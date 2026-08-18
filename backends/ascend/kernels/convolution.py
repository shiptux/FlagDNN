# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

"""Ascend-owned strided 1D/2D/3D convolution FProp kernel."""

import triton
import triton.language as tl


@triton.jit
def _convolution_fprop_2d_nchw_tiled(
    input_ptr,
    filter_ptr,
    output_ptr,
    GROUPS: tl.constexpr,
    OUTPUT_CHANNELS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    INPUT_DIM_0: tl.constexpr,
    INPUT_DIM_3: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    FILTER_DIM_3: tl.constexpr,
    FILTER_DIM_4: tl.constexpr,
    FILTER_STRIDE_0: tl.constexpr,
    FILTER_STRIDE_1: tl.constexpr,
    FILTER_STRIDE_3: tl.constexpr,
    FILTER_STRIDE_4: tl.constexpr,
    OUTPUT_DIM_3: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    PRE_PADDING_1: tl.constexpr,
    PRE_PADDING_2: tl.constexpr,
    CONV_STRIDE_1: tl.constexpr,
    CONV_STRIDE_2: tl.constexpr,
    DILATION_1: tl.constexpr,
    DILATION_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    block_m: tl.constexpr = BLOCK_SIZE // 32
    block_n: tl.constexpr = BLOCK_SIZE // 32
    block_k: tl.constexpr = 8
    output_area: tl.constexpr = OUTPUT_DIM_3 * OUTPUT_DIM_4
    output_positions: tl.constexpr = INPUT_DIM_0 * output_area
    output_channels_per_group: tl.constexpr = OUTPUT_CHANNELS // GROUPS
    filter_area: tl.constexpr = FILTER_DIM_3 * FILTER_DIM_4
    reduction_extent: tl.constexpr = CHANNELS_PER_GROUP * filter_area
    tiles_m: tl.constexpr = tl.cdiv(output_positions, block_m)
    tiles_n: tl.constexpr = tl.cdiv(output_channels_per_group, block_n)
    tiles_per_group: tl.constexpr = tiles_m * tiles_n
    total_tasks: tl.constexpr = GROUPS * tiles_per_group

    task = tl.program_id(0).to(tl.int64)
    task_stride = tl.num_programs(0)
    while task < total_tasks:
        group = task // tiles_per_group
        tile = task % tiles_per_group
        tile_m = tile // tiles_n
        tile_n = tile % tiles_n

        rows = tile_m * block_m + tl.arange(0, block_m)
        row_mask = rows < output_positions
        safe_rows = tl.where(row_mask, rows, 0).to(tl.int64)
        batch = safe_rows // output_area
        output_hw = safe_rows % output_area
        output_h = output_hw // OUTPUT_DIM_4
        output_w = output_hw % OUTPUT_DIM_4

        group_output_channels = tile_n * block_n + tl.arange(0, block_n)
        channel_mask = group_output_channels < output_channels_per_group
        safe_group_output_channels = tl.where(
            channel_mask, group_output_channels, 0
        )
        output_channels = (
            group * output_channels_per_group + safe_group_output_channels
        )

        reduction_base = tl.arange(0, block_k)
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        for reduction_start in range(0, reduction_extent, block_k):
            reduction = reduction_start + reduction_base
            reduction_mask = reduction < reduction_extent
            safe_reduction = tl.where(reduction_mask, reduction, 0)
            input_channel = safe_reduction // filter_area
            filter_hw = safe_reduction % filter_area
            filter_h = filter_hw // FILTER_DIM_4
            filter_w = filter_hw % FILTER_DIM_4

            input_h = (
                output_h[:, None] * CONV_STRIDE_1
                - PRE_PADDING_1
                + filter_h[None, :] * DILATION_1
            )
            input_w = (
                output_w[:, None] * CONV_STRIDE_2
                - PRE_PADDING_2
                + filter_w[None, :] * DILATION_2
            )
            valid_h = (input_h >= 0) & (input_h < INPUT_DIM_3)
            valid_w = (input_w >= 0) & (input_w < INPUT_DIM_4)
            safe_input_h = tl.where(valid_h, input_h, 0)
            safe_input_w = tl.where(valid_w, input_w, 0)
            input_values = tl.load(
                input_ptr
                + batch[:, None] * INPUT_STRIDE_0
                + (
                    group * CHANNELS_PER_GROUP
                    + input_channel[None, :]
                )
                * INPUT_STRIDE_1
                + safe_input_h * INPUT_STRIDE_3
                + safe_input_w * INPUT_STRIDE_4,
                mask=(
                    row_mask[:, None]
                    & reduction_mask[None, :]
                    & valid_h
                    & valid_w
                ),
                other=0.0,
            )
            weights = tl.load(
                filter_ptr
                + output_channels[None, :] * FILTER_STRIDE_0
                + input_channel[:, None] * FILTER_STRIDE_1
                + filter_h[:, None] * FILTER_STRIDE_3
                + filter_w[:, None] * FILTER_STRIDE_4,
                mask=reduction_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            accumulator += tl.sum(
                input_values.to(tl.float32)[:, :, None]
                * weights.to(tl.float32)[None, :, :],
                axis=1,
            )

        output_offsets = (
            batch[:, None] * OUTPUT_STRIDE_0
            + output_channels[None, :] * OUTPUT_STRIDE_1
            + output_h[:, None] * OUTPUT_STRIDE_3
            + output_w[:, None] * OUTPUT_STRIDE_4
        )
        tl.store(
            output_ptr + output_offsets,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=row_mask[:, None] & channel_mask[None, :],
        )
        task += task_stride


@triton.jit
def convolution_fprop_persistent_kernel(
    input_ptr,
    filter_ptr,
    output_ptr,
    n_elements,
    SPATIAL_RANK: tl.constexpr,
    GROUPS: tl.constexpr,
    INPUT_CHANNELS: tl.constexpr,
    OUTPUT_CHANNELS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    INPUT_DIM_0: tl.constexpr,
    INPUT_DIM_1: tl.constexpr,
    INPUT_DIM_2: tl.constexpr,
    INPUT_DIM_3: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_2: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    FILTER_DIM_0: tl.constexpr,
    FILTER_DIM_1: tl.constexpr,
    FILTER_DIM_2: tl.constexpr,
    FILTER_DIM_3: tl.constexpr,
    FILTER_DIM_4: tl.constexpr,
    FILTER_STRIDE_0: tl.constexpr,
    FILTER_STRIDE_1: tl.constexpr,
    FILTER_STRIDE_2: tl.constexpr,
    FILTER_STRIDE_3: tl.constexpr,
    FILTER_STRIDE_4: tl.constexpr,
    OUTPUT_DIM_0: tl.constexpr,
    OUTPUT_DIM_1: tl.constexpr,
    OUTPUT_DIM_2: tl.constexpr,
    OUTPUT_DIM_3: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_2: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    PRE_PADDING_0: tl.constexpr,
    PRE_PADDING_1: tl.constexpr,
    PRE_PADDING_2: tl.constexpr,
    POST_PADDING_0: tl.constexpr,
    POST_PADDING_1: tl.constexpr,
    POST_PADDING_2: tl.constexpr,
    CONV_STRIDE_0: tl.constexpr,
    CONV_STRIDE_1: tl.constexpr,
    CONV_STRIDE_2: tl.constexpr,
    DILATION_0: tl.constexpr,
    DILATION_1: tl.constexpr,
    DILATION_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    dense_nchw_2d: tl.constexpr = (
        SPATIAL_RANK == 2
        and INPUT_STRIDE_4 == 1
        and INPUT_STRIDE_3 == INPUT_DIM_4
        and INPUT_STRIDE_1 == INPUT_DIM_3 * INPUT_STRIDE_3
        and INPUT_STRIDE_0 == INPUT_CHANNELS * INPUT_STRIDE_1
        and FILTER_STRIDE_4 == 1
        and FILTER_STRIDE_3 == FILTER_DIM_4
        and FILTER_STRIDE_1 == FILTER_DIM_3 * FILTER_STRIDE_3
        and FILTER_STRIDE_0 == CHANNELS_PER_GROUP * FILTER_STRIDE_1
        and OUTPUT_STRIDE_4 == 1
        and OUTPUT_STRIDE_3 == OUTPUT_DIM_4
        and OUTPUT_STRIDE_1 == OUTPUT_DIM_3 * OUTPUT_STRIDE_3
        and OUTPUT_STRIDE_0 == OUTPUT_CHANNELS * OUTPUT_STRIDE_1
    )
    if dense_nchw_2d:
        _convolution_fprop_2d_nchw_tiled(
            input_ptr,
            filter_ptr,
            output_ptr,
            GROUPS,
            OUTPUT_CHANNELS,
            CHANNELS_PER_GROUP,
            INPUT_DIM_0,
            INPUT_DIM_3,
            INPUT_DIM_4,
            INPUT_STRIDE_0,
            INPUT_STRIDE_1,
            INPUT_STRIDE_3,
            INPUT_STRIDE_4,
            FILTER_DIM_3,
            FILTER_DIM_4,
            FILTER_STRIDE_0,
            FILTER_STRIDE_1,
            FILTER_STRIDE_3,
            FILTER_STRIDE_4,
            OUTPUT_DIM_3,
            OUTPUT_DIM_4,
            OUTPUT_STRIDE_0,
            OUTPUT_STRIDE_1,
            OUTPUT_STRIDE_3,
            OUTPUT_STRIDE_4,
            PRE_PADDING_1,
            PRE_PADDING_2,
            CONV_STRIDE_1,
            CONV_STRIDE_2,
            DILATION_1,
            DILATION_2,
            BLOCK_SIZE,
            WORKER_COUNT,
        )
        return

    program = tl.program_id(0)
    tile_count = tl.cdiv(n_elements, BLOCK_SIZE)
    tile = program
    while tile < tile_count:
        logical = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        logical_mask = logical < n_elements
        remaining = logical.to(tl.int64)
        output_w = remaining % OUTPUT_DIM_4
        remaining //= OUTPUT_DIM_4
        output_h = remaining % OUTPUT_DIM_3
        remaining //= OUTPUT_DIM_3
        output_d = remaining % OUTPUT_DIM_2
        remaining //= OUTPUT_DIM_2
        output_channel = remaining % OUTPUT_DIM_1
        output_batch = remaining // OUTPUT_DIM_1
        safe_output_batch = tl.where(logical_mask, output_batch, 0)

        output_channels_per_group = OUTPUT_CHANNELS // GROUPS
        group = output_channel // output_channels_per_group
        input_channel_base = group * CHANNELS_PER_GROUP
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for channel_offset in range(0, CHANNELS_PER_GROUP):
            input_channel = input_channel_base + channel_offset
            for filter_d in range(0, FILTER_DIM_2):
                input_d = (
                    output_d * CONV_STRIDE_0
                    - PRE_PADDING_0
                    + filter_d * DILATION_0
                )
                for filter_h in range(0, FILTER_DIM_3):
                    input_h = (
                        output_h * CONV_STRIDE_1
                        - PRE_PADDING_1
                        + filter_h * DILATION_1
                    )
                    for filter_w in range(0, FILTER_DIM_4):
                        input_w = (
                            output_w * CONV_STRIDE_2
                            - PRE_PADDING_2
                            + filter_w * DILATION_2
                        )
                        valid_d = (input_d >= 0) & (input_d < INPUT_DIM_2)
                        valid_h = (input_h >= 0) & (input_h < INPUT_DIM_3)
                        valid_w = (input_w >= 0) & (input_w < INPUT_DIM_4)
                        input_mask = (
                            logical_mask & valid_d & valid_h & valid_w
                        )
                        # Ascend may still evaluate masked pointer arithmetic.
                        # Keep every lane's address within its allocation and
                        # use input_mask solely to provide padding zeros.
                        safe_input_d = tl.where(valid_d, input_d, 0)
                        safe_input_h = tl.where(valid_h, input_h, 0)
                        safe_input_w = tl.where(valid_w, input_w, 0)
                        input_offset = (
                            safe_output_batch * INPUT_STRIDE_0
                            + input_channel * INPUT_STRIDE_1
                            + safe_input_d * INPUT_STRIDE_2
                            + safe_input_h * INPUT_STRIDE_3
                            + safe_input_w * INPUT_STRIDE_4
                        )
                        filter_offset = (
                            output_channel * FILTER_STRIDE_0
                            + channel_offset * FILTER_STRIDE_1
                            + filter_d * FILTER_STRIDE_2
                            + filter_h * FILTER_STRIDE_3
                            + filter_w * FILTER_STRIDE_4
                        )
                        input_value = tl.load(
                            input_ptr + input_offset,
                            mask=input_mask,
                            other=0.0,
                        ).to(tl.float32)
                        filter_value = tl.load(
                            filter_ptr + filter_offset,
                            mask=logical_mask,
                            other=0.0,
                        ).to(tl.float32)
                        accumulator += input_value * filter_value

        output_offset = (
            safe_output_batch * OUTPUT_STRIDE_0
            + output_channel * OUTPUT_STRIDE_1
            + output_d * OUTPUT_STRIDE_2
            + output_h * OUTPUT_STRIDE_3
            + output_w * OUTPUT_STRIDE_4
        )
        tl.store(
            output_ptr + output_offset,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=logical_mask,
        )
        tile += WORKER_COUNT
