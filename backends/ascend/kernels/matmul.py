# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

"""Ascend-owned strided and broadcasted MatMul kernel."""

import triton
import triton.language as tl


@triton.jit
def matmul_strided_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    A_BATCH_STRIDE_0: tl.constexpr,
    A_BATCH_STRIDE_1: tl.constexpr,
    A_BATCH_STRIDE_2: tl.constexpr,
    A_BATCH_STRIDE_3: tl.constexpr,
    A_BATCH_STRIDE_4: tl.constexpr,
    A_BATCH_STRIDE_5: tl.constexpr,
    B_BATCH_STRIDE_0: tl.constexpr,
    B_BATCH_STRIDE_1: tl.constexpr,
    B_BATCH_STRIDE_2: tl.constexpr,
    B_BATCH_STRIDE_3: tl.constexpr,
    B_BATCH_STRIDE_4: tl.constexpr,
    B_BATCH_STRIDE_5: tl.constexpr,
    C_BATCH_STRIDE_0: tl.constexpr,
    C_BATCH_STRIDE_1: tl.constexpr,
    C_BATCH_STRIDE_2: tl.constexpr,
    C_BATCH_STRIDE_3: tl.constexpr,
    C_BATCH_STRIDE_4: tl.constexpr,
    C_BATCH_STRIDE_5: tl.constexpr,
    A_STRIDE_M: tl.constexpr,
    A_STRIDE_K: tl.constexpr,
    B_STRIDE_K: tl.constexpr,
    B_STRIDE_N: tl.constexpr,
    C_STRIDE_M: tl.constexpr,
    C_STRIDE_N: tl.constexpr,
    INPUT_IS_FLOAT32: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    tiles_m = tl.cdiv(M, BLOCK_SIZE)
    tiles_n = tl.cdiv(N, BLOCK_SIZE)
    tiles_per_batch = tiles_m * tiles_n
    total_tasks = BATCH * tiles_per_batch
    task = tl.program_id(0).to(tl.int64)
    while task < total_tasks:
        batch = task // tiles_per_batch
        tile = task % tiles_per_batch
        if GROUP_M == 1:
            tile_m = tile // tiles_n
            tile_n = tile % tiles_n
        elif GROUP_M >= tiles_m:
            tile_m = tile % tiles_m
            tile_n = tile // tiles_m
        else:
            tiles_per_group = GROUP_M * tiles_n
            group = tile // tiles_per_group
            first_tile_m = group * GROUP_M
            group_m = tl.minimum(tiles_m - first_tile_m, GROUP_M)
            tile_in_group = tile % tiles_per_group
            tile_m = first_tile_m + tile_in_group % group_m
            tile_n = tile_in_group // group_m

        remaining = batch
        a_batch_offset = tl.zeros((), dtype=tl.int64)
        b_batch_offset = tl.zeros((), dtype=tl.int64)
        c_batch_offset = tl.zeros((), dtype=tl.int64)
        coordinate = remaining % DIM_5
        remaining //= DIM_5
        a_batch_offset += coordinate * A_BATCH_STRIDE_5
        b_batch_offset += coordinate * B_BATCH_STRIDE_5
        c_batch_offset += coordinate * C_BATCH_STRIDE_5
        coordinate = remaining % DIM_4
        remaining //= DIM_4
        a_batch_offset += coordinate * A_BATCH_STRIDE_4
        b_batch_offset += coordinate * B_BATCH_STRIDE_4
        c_batch_offset += coordinate * C_BATCH_STRIDE_4
        coordinate = remaining % DIM_3
        remaining //= DIM_3
        a_batch_offset += coordinate * A_BATCH_STRIDE_3
        b_batch_offset += coordinate * B_BATCH_STRIDE_3
        c_batch_offset += coordinate * C_BATCH_STRIDE_3
        coordinate = remaining % DIM_2
        remaining //= DIM_2
        a_batch_offset += coordinate * A_BATCH_STRIDE_2
        b_batch_offset += coordinate * B_BATCH_STRIDE_2
        c_batch_offset += coordinate * C_BATCH_STRIDE_2
        coordinate = remaining % DIM_1
        remaining //= DIM_1
        a_batch_offset += coordinate * A_BATCH_STRIDE_1
        b_batch_offset += coordinate * B_BATCH_STRIDE_1
        c_batch_offset += coordinate * C_BATCH_STRIDE_1
        coordinate = remaining % DIM_0
        a_batch_offset += coordinate * A_BATCH_STRIDE_0
        b_batch_offset += coordinate * B_BATCH_STRIDE_0
        c_batch_offset += coordinate * C_BATCH_STRIDE_0

        rows = tile_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        columns = tile_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        reduction = tl.arange(0, BLOCK_SIZE)
        a_tile_ptrs = (
            a_ptr
            + a_batch_offset
            + rows[:, None] * A_STRIDE_M
            + reduction[None, :] * A_STRIDE_K
        )
        b_tile_ptrs = (
            b_ptr
            + b_batch_offset
            + reduction[:, None] * B_STRIDE_K
            + columns[None, :] * B_STRIDE_N
        )
        accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        for reduction_start in range(0, K, BLOCK_SIZE):
            reduction_offsets = reduction_start + reduction
            a = tl.load(
                a_tile_ptrs,
                mask=(rows[:, None] < M) & (reduction_offsets[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_tile_ptrs,
                mask=(reduction_offsets[:, None] < K)
                & (columns[None, :] < N),
                other=0.0,
            )
            if INPUT_IS_FLOAT32:
                accumulator += tl.sum(
                    a[:, :, None] * b[None, :, :], axis=1
                )
            else:
                accumulator += tl.dot(a, b)
            a_tile_ptrs += BLOCK_SIZE * A_STRIDE_K
            b_tile_ptrs += BLOCK_SIZE * B_STRIDE_K

        output_offsets = (
            c_batch_offset
            + rows[:, None] * C_STRIDE_M
            + columns[None, :] * C_STRIDE_N
        )
        output_mask = (rows[:, None] < M) & (columns[None, :] < N)
        tl.store(
            output_ptr + output_offsets,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=output_mask,
        )
        task += WORKER_COUNT
