# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Ascend-owned persistent single-axis reduction kernels."""

import triton
import triton.language as tl


REDUCTION_SUM = tl.constexpr(0)
REDUCTION_AVG = tl.constexpr(1)
REDUCTION_MUL = tl.constexpr(2)


@triton.jit
def _multiply(left, right):
    return left * right


@triton.jit
def _combine(accumulator, values, REDUCTION_MODE: tl.constexpr):
    if REDUCTION_MODE == REDUCTION_MUL:
        return accumulator * tl.reduce(values, axis=0, combine_fn=_multiply)
    return accumulator + tl.sum(values, axis=0)


@triton.jit
def reduction_3d_persistent_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    RANK: tl.constexpr,
    OUTPUT_RANK: tl.constexpr,
    AXIS: tl.constexpr,
    KEEP_DIMENSIONS: tl.constexpr,
    OUTER: tl.constexpr,
    REDUCTION_SIZE: tl.constexpr,
    INNER: tl.constexpr,
    OUTPUT_ELEMENTS: tl.constexpr,
    REDUCTION_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    output_index = tl.program_id(0).to(tl.int64)
    while output_index < n_elements:
        outer_index = output_index // INNER
        inner_index = output_index % INNER
        input_base = outer_index * REDUCTION_SIZE * INNER + inner_index
        if REDUCTION_MODE == REDUCTION_MUL:
            accumulator = 1.0
        else:
            accumulator = 0.0
        reduction_start = output_index * 0
        while reduction_start < REDUCTION_SIZE:
            reduction_offsets = reduction_start + tl.arange(
                0, BLOCK_SIZE
            ).to(tl.int64)
            active = reduction_offsets < REDUCTION_SIZE
            values = tl.load(
                input_ptr + input_base + reduction_offsets * INNER,
                mask=active,
                other=1.0 if REDUCTION_MODE == REDUCTION_MUL else 0.0,
            ).to(tl.float32)
            accumulator = _combine(accumulator, values, REDUCTION_MODE)
            reduction_start += BLOCK_SIZE
        if REDUCTION_MODE == REDUCTION_AVG:
            accumulator /= REDUCTION_SIZE
        tl.store(
            output_ptr + output_index,
            accumulator.to(output_ptr.dtype.element_ty),
        )
        output_index += WORKER_COUNT


@triton.jit
def reduction_strided_persistent_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    RANK: tl.constexpr,
    OUTPUT_RANK: tl.constexpr,
    AXIS: tl.constexpr,
    KEEP_DIMENSIONS: tl.constexpr,
    OUTER: tl.constexpr,
    REDUCTION_SIZE: tl.constexpr,
    INNER: tl.constexpr,
    OUTPUT_ELEMENTS: tl.constexpr,
    REDUCTION_MODE: tl.constexpr,
    INPUT_DIM_0: tl.constexpr,
    INPUT_DIM_1: tl.constexpr,
    INPUT_DIM_2: tl.constexpr,
    INPUT_DIM_3: tl.constexpr,
    INPUT_DIM_4: tl.constexpr,
    INPUT_DIM_5: tl.constexpr,
    INPUT_DIM_6: tl.constexpr,
    INPUT_DIM_7: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_2: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    INPUT_STRIDE_5: tl.constexpr,
    INPUT_STRIDE_6: tl.constexpr,
    INPUT_STRIDE_7: tl.constexpr,
    OUTPUT_DIM_0: tl.constexpr,
    OUTPUT_DIM_1: tl.constexpr,
    OUTPUT_DIM_2: tl.constexpr,
    OUTPUT_DIM_3: tl.constexpr,
    OUTPUT_DIM_4: tl.constexpr,
    OUTPUT_DIM_5: tl.constexpr,
    OUTPUT_DIM_6: tl.constexpr,
    OUTPUT_DIM_7: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_2: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    OUTPUT_STRIDE_5: tl.constexpr,
    OUTPUT_STRIDE_6: tl.constexpr,
    OUTPUT_STRIDE_7: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    output_index = tl.program_id(0).to(tl.int64)
    while output_index < n_elements:
        output_remaining = output_index
        output_offset = output_index * 0
        coordinate = output_remaining % OUTPUT_DIM_7
        output_remaining //= OUTPUT_DIM_7
        output_offset += coordinate * OUTPUT_STRIDE_7
        coordinate = output_remaining % OUTPUT_DIM_6
        output_remaining //= OUTPUT_DIM_6
        output_offset += coordinate * OUTPUT_STRIDE_6
        coordinate = output_remaining % OUTPUT_DIM_5
        output_remaining //= OUTPUT_DIM_5
        output_offset += coordinate * OUTPUT_STRIDE_5
        coordinate = output_remaining % OUTPUT_DIM_4
        output_remaining //= OUTPUT_DIM_4
        output_offset += coordinate * OUTPUT_STRIDE_4
        coordinate = output_remaining % OUTPUT_DIM_3
        output_remaining //= OUTPUT_DIM_3
        output_offset += coordinate * OUTPUT_STRIDE_3
        coordinate = output_remaining % OUTPUT_DIM_2
        output_remaining //= OUTPUT_DIM_2
        output_offset += coordinate * OUTPUT_STRIDE_2
        coordinate = output_remaining % OUTPUT_DIM_1
        output_remaining //= OUTPUT_DIM_1
        output_offset += coordinate * OUTPUT_STRIDE_1
        output_offset += (output_remaining % OUTPUT_DIM_0) * OUTPUT_STRIDE_0

        reduction_start = output_index * 0
        if REDUCTION_MODE == REDUCTION_MUL:
            accumulator = 1.0
        else:
            accumulator = 0.0
        while reduction_start < REDUCTION_SIZE:
            reduction_offsets = reduction_start + tl.arange(
                0, BLOCK_SIZE
            ).to(tl.int64)
            active = reduction_offsets < REDUCTION_SIZE
            input_linear = (
                (output_index // INNER) * REDUCTION_SIZE * INNER
                + reduction_offsets * INNER
                + output_index % INNER
            )
            input_remaining = input_linear
            input_offset = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
            input_coordinate = input_remaining % INPUT_DIM_7
            input_remaining //= INPUT_DIM_7
            input_offset += input_coordinate * INPUT_STRIDE_7
            input_coordinate = input_remaining % INPUT_DIM_6
            input_remaining //= INPUT_DIM_6
            input_offset += input_coordinate * INPUT_STRIDE_6
            input_coordinate = input_remaining % INPUT_DIM_5
            input_remaining //= INPUT_DIM_5
            input_offset += input_coordinate * INPUT_STRIDE_5
            input_coordinate = input_remaining % INPUT_DIM_4
            input_remaining //= INPUT_DIM_4
            input_offset += input_coordinate * INPUT_STRIDE_4
            input_coordinate = input_remaining % INPUT_DIM_3
            input_remaining //= INPUT_DIM_3
            input_offset += input_coordinate * INPUT_STRIDE_3
            input_coordinate = input_remaining % INPUT_DIM_2
            input_remaining //= INPUT_DIM_2
            input_offset += input_coordinate * INPUT_STRIDE_2
            input_coordinate = input_remaining % INPUT_DIM_1
            input_remaining //= INPUT_DIM_1
            input_offset += input_coordinate * INPUT_STRIDE_1
            input_coordinate = input_remaining % INPUT_DIM_0
            input_offset += input_coordinate * INPUT_STRIDE_0
            values = tl.load(
                input_ptr + input_offset,
                mask=active,
                other=1.0 if REDUCTION_MODE == REDUCTION_MUL else 0.0,
            ).to(tl.float32)
            accumulator = _combine(accumulator, values, REDUCTION_MODE)
            reduction_start += BLOCK_SIZE
        if REDUCTION_MODE == REDUCTION_AVG:
            accumulator /= REDUCTION_SIZE
        tl.store(
            output_ptr + output_offset,
            accumulator.to(output_ptr.dtype.element_ty),
        )
        output_index += WORKER_COUNT
