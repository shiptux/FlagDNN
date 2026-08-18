# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Ascend-owned persistent normalization kernels."""

import triton
import triton.language as tl


@triton.jit
def _batchnorm_training_tensor_offset(
    logical_index,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    STRIDE_2: tl.constexpr,
    STRIDE_3: tl.constexpr,
    STRIDE_4: tl.constexpr,
    STRIDE_5: tl.constexpr,
    STRIDE_6: tl.constexpr,
    STRIDE_7: tl.constexpr,
):
    remaining = logical_index
    offset = logical_index * 0
    coordinate = remaining % DIM_7
    remaining //= DIM_7
    offset += coordinate * STRIDE_7
    coordinate = remaining % DIM_6
    remaining //= DIM_6
    offset += coordinate * STRIDE_6
    coordinate = remaining % DIM_5
    remaining //= DIM_5
    offset += coordinate * STRIDE_5
    coordinate = remaining % DIM_4
    remaining //= DIM_4
    offset += coordinate * STRIDE_4
    coordinate = remaining % DIM_3
    remaining //= DIM_3
    offset += coordinate * STRIDE_3
    coordinate = remaining % DIM_2
    remaining //= DIM_2
    offset += coordinate * STRIDE_2
    coordinate = remaining % DIM_1
    remaining //= DIM_1
    offset += coordinate * STRIDE_1
    coordinate = remaining % DIM_0
    offset += coordinate * STRIDE_0
    return offset


@triton.jit
def batchnorm_training_persistent_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    previous_running_mean_ptr,
    previous_running_variance_ptr,
    y_ptr,
    mean_ptr,
    inv_variance_ptr,
    next_running_mean_ptr,
    next_running_variance_ptr,
    n_elements,
    RANK: tl.constexpr,
    BATCH: tl.constexpr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    REDUCTION_ELEMENTS: tl.constexpr,
    EPSILON: tl.constexpr,
    MOMENTUM: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    X_STRIDE_3: tl.constexpr,
    X_STRIDE_4: tl.constexpr,
    X_STRIDE_5: tl.constexpr,
    X_STRIDE_6: tl.constexpr,
    X_STRIDE_7: tl.constexpr,
    Y_STRIDE_0: tl.constexpr,
    Y_STRIDE_1: tl.constexpr,
    Y_STRIDE_2: tl.constexpr,
    Y_STRIDE_3: tl.constexpr,
    Y_STRIDE_4: tl.constexpr,
    Y_STRIDE_5: tl.constexpr,
    Y_STRIDE_6: tl.constexpr,
    Y_STRIDE_7: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    channel = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    while channel < CHANNELS:
        value_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        reduction_base = 0
        while reduction_base < REDUCTION_ELEMENTS:
            reduction_index = reduction_base + lane
            mask = reduction_index < REDUCTION_ELEMENTS
            logical_index = (
                (reduction_index // SPATIAL) * CHANNELS * SPATIAL
                + channel * SPATIAL
                + reduction_index % SPATIAL
            )
            x_offset = _batchnorm_training_tensor_offset(
                logical_index,
                DIM_0,
                DIM_1,
                DIM_2,
                DIM_3,
                DIM_4,
                DIM_5,
                DIM_6,
                DIM_7,
                X_STRIDE_0,
                X_STRIDE_1,
                X_STRIDE_2,
                X_STRIDE_3,
                X_STRIDE_4,
                X_STRIDE_5,
                X_STRIDE_6,
                X_STRIDE_7,
            )
            value_sum += tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(
                tl.float32
            )
            reduction_base += BLOCK_SIZE
        channel_mean = tl.sum(value_sum, axis=0) / REDUCTION_ELEMENTS

        square_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        reduction_base = 0
        while reduction_base < REDUCTION_ELEMENTS:
            reduction_index = reduction_base + lane
            mask = reduction_index < REDUCTION_ELEMENTS
            logical_index = (
                (reduction_index // SPATIAL) * CHANNELS * SPATIAL
                + channel * SPATIAL
                + reduction_index % SPATIAL
            )
            x_offset = _batchnorm_training_tensor_offset(
                logical_index,
                DIM_0,
                DIM_1,
                DIM_2,
                DIM_3,
                DIM_4,
                DIM_5,
                DIM_6,
                DIM_7,
                X_STRIDE_0,
                X_STRIDE_1,
                X_STRIDE_2,
                X_STRIDE_3,
                X_STRIDE_4,
                X_STRIDE_5,
                X_STRIDE_6,
                X_STRIDE_7,
            )
            value = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(
                tl.float32
            )
            centered = tl.where(mask, value - channel_mean, 0.0)
            square_sum += centered * centered
            reduction_base += BLOCK_SIZE
        variance = tl.sum(square_sum, axis=0) / REDUCTION_ELEMENTS
        variance_with_epsilon = variance + EPSILON
        channel_inv_variance = tl.rsqrt(variance_with_epsilon)
        channel_inv_variance *= 1.5 - (
            0.5
            * variance_with_epsilon
            * channel_inv_variance
            * channel_inv_variance
        )
        tl.store(mean_ptr + channel, channel_mean)
        tl.store(inv_variance_ptr + channel, channel_inv_variance)

        previous_mean = tl.load(previous_running_mean_ptr + channel).to(
            tl.float32
        )
        previous_variance = tl.load(
            previous_running_variance_ptr + channel
        ).to(tl.float32)
        unbiased_variance = tl.where(
            REDUCTION_ELEMENTS > 1,
            variance * REDUCTION_ELEMENTS / (REDUCTION_ELEMENTS - 1),
            variance,
        )
        tl.store(
            next_running_mean_ptr + channel,
            previous_mean * (1.0 - MOMENTUM) + channel_mean * MOMENTUM,
        )
        tl.store(
            next_running_variance_ptr + channel,
            previous_variance * (1.0 - MOMENTUM)
            + unbiased_variance * MOMENTUM,
        )

        channel_scale = tl.load(scale_ptr + channel).to(tl.float32)
        channel_bias = tl.load(bias_ptr + channel).to(tl.float32)
        reduction_base = 0
        while reduction_base < REDUCTION_ELEMENTS:
            reduction_index = reduction_base + lane
            mask = reduction_index < REDUCTION_ELEMENTS
            logical_index = (
                (reduction_index // SPATIAL) * CHANNELS * SPATIAL
                + channel * SPATIAL
                + reduction_index % SPATIAL
            )
            x_offset = _batchnorm_training_tensor_offset(
                logical_index,
                DIM_0,
                DIM_1,
                DIM_2,
                DIM_3,
                DIM_4,
                DIM_5,
                DIM_6,
                DIM_7,
                X_STRIDE_0,
                X_STRIDE_1,
                X_STRIDE_2,
                X_STRIDE_3,
                X_STRIDE_4,
                X_STRIDE_5,
                X_STRIDE_6,
                X_STRIDE_7,
            )
            y_offset = _batchnorm_training_tensor_offset(
                logical_index,
                DIM_0,
                DIM_1,
                DIM_2,
                DIM_3,
                DIM_4,
                DIM_5,
                DIM_6,
                DIM_7,
                Y_STRIDE_0,
                Y_STRIDE_1,
                Y_STRIDE_2,
                Y_STRIDE_3,
                Y_STRIDE_4,
                Y_STRIDE_5,
                Y_STRIDE_6,
                Y_STRIDE_7,
            )
            value = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(
                tl.float32
            )
            result = (
                (value - channel_mean)
                * channel_inv_variance
                * channel_scale
                + channel_bias
            )
            tl.store(
                y_ptr + y_offset,
                result.to(y_ptr.dtype.element_ty),
                mask=mask,
            )
            reduction_base += BLOCK_SIZE
        channel += WORKER_COUNT


@triton.jit
def layernorm_persistent_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    mean_ptr,
    inv_variance_ptr,
    n_elements,
    ROWS: tl.constexpr,
    NORMALIZED_ELEMENTS: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    while row < ROWS:
        value_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        normalized_base = 0
        while normalized_base < NORMALIZED_ELEMENTS:
            normalized_index = lane + normalized_base
            linear_index = row * NORMALIZED_ELEMENTS + normalized_index
            mask = (normalized_index < NORMALIZED_ELEMENTS) & (
                linear_index < n_elements
            )
            value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(
                tl.float32
            )
            value_sum += value
            normalized_base += BLOCK_SIZE
        mean = tl.sum(value_sum, axis=0) / NORMALIZED_ELEMENTS
        tl.store(mean_ptr + row, mean)

        square_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        normalized_base = 0
        while normalized_base < NORMALIZED_ELEMENTS:
            normalized_index = lane + normalized_base
            linear_index = row * NORMALIZED_ELEMENTS + normalized_index
            mask = (normalized_index < NORMALIZED_ELEMENTS) & (
                linear_index < n_elements
            )
            value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(
                tl.float32
            )
            centered = tl.where(mask, value - mean, 0.0)
            square_sum += centered * centered
            normalized_base += BLOCK_SIZE
        inv_variance = tl.rsqrt(
            tl.sum(square_sum, axis=0) / NORMALIZED_ELEMENTS + EPSILON
        )
        tl.store(inv_variance_ptr + row, inv_variance)

        normalized_base = 0
        while normalized_base < NORMALIZED_ELEMENTS:
            normalized_index = lane + normalized_base
            linear_index = row * NORMALIZED_ELEMENTS + normalized_index
            mask = (normalized_index < NORMALIZED_ELEMENTS) & (
                linear_index < n_elements
            )
            value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(
                tl.float32
            )
            scale = tl.load(
                scale_ptr + normalized_index, mask=mask, other=0.0
            ).to(tl.float32)
            bias = tl.load(
                bias_ptr + normalized_index, mask=mask, other=0.0
            ).to(tl.float32)
            result = (value - mean) * inv_variance * scale + bias
            tl.store(
                y_ptr + linear_index,
                result.to(y_ptr.dtype.element_ty),
                mask=mask,
            )
            normalized_base += BLOCK_SIZE
        row += WORKER_COUNT


@triton.jit
def rmsnorm_persistent_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    inv_variance_ptr,
    n_elements,
    ROWS: tl.constexpr,
    NORMALIZED_ELEMENTS: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    while row < ROWS:
        square_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        normalized_base = 0
        while normalized_base < NORMALIZED_ELEMENTS:
            normalized_index = lane + normalized_base
            linear_index = row * NORMALIZED_ELEMENTS + normalized_index
            mask = (normalized_index < NORMALIZED_ELEMENTS) & (
                linear_index < n_elements
            )
            value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(
                tl.float32
            )
            square_sum += value * value
            normalized_base += BLOCK_SIZE
        inv_variance = tl.rsqrt(
            tl.sum(square_sum, axis=0) / NORMALIZED_ELEMENTS + EPSILON
        )
        tl.store(inv_variance_ptr + row, inv_variance)

        normalized_base = 0
        while normalized_base < NORMALIZED_ELEMENTS:
            normalized_index = lane + normalized_base
            linear_index = row * NORMALIZED_ELEMENTS + normalized_index
            mask = (normalized_index < NORMALIZED_ELEMENTS) & (
                linear_index < n_elements
            )
            value = tl.load(x_ptr + linear_index, mask=mask, other=0.0).to(
                tl.float32
            )
            scale = tl.load(
                scale_ptr + normalized_index, mask=mask, other=0.0
            ).to(tl.float32)
            bias = tl.load(
                bias_ptr + normalized_index, mask=mask, other=0.0
            ).to(tl.float32)
            result = value * inv_variance * scale + bias
            tl.store(
                y_ptr + linear_index,
                result.to(y_ptr.dtype.element_ty),
                mask=mask,
            )
            normalized_base += BLOCK_SIZE
        row += WORKER_COUNT


@triton.jit
def batchnorm_inference_nchw_persistent_kernel(
    x_ptr,
    mean_ptr,
    inv_variance_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    n_elements,
    RANK: tl.constexpr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    block_base = program_id * BLOCK_SIZE
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    while block_base < n_elements:
        logical_index = block_base + lane
        mask = logical_index < n_elements
        channel = (logical_index // SPATIAL) % CHANNELS
        value = tl.load(x_ptr + logical_index, mask=mask, other=0.0).to(
            tl.float32
        )
        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        inv_variance = tl.load(
            inv_variance_ptr + channel, mask=mask, other=0.0
        ).to(tl.float32)
        scale = tl.load(scale_ptr + channel, mask=mask, other=0.0).to(
            tl.float32
        )
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        result = (value - mean) * inv_variance * scale + bias
        tl.store(
            y_ptr + logical_index,
            result.to(y_ptr.dtype.element_ty),
            mask=mask,
        )
        block_base += BLOCK_SIZE * WORKER_COUNT


@triton.jit
def batchnorm_inference_strided_persistent_kernel(
    x_ptr,
    mean_ptr,
    inv_variance_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    n_elements,
    RANK: tl.constexpr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    X_STRIDE_3: tl.constexpr,
    X_STRIDE_4: tl.constexpr,
    X_STRIDE_5: tl.constexpr,
    X_STRIDE_6: tl.constexpr,
    X_STRIDE_7: tl.constexpr,
    Y_STRIDE_0: tl.constexpr,
    Y_STRIDE_1: tl.constexpr,
    Y_STRIDE_2: tl.constexpr,
    Y_STRIDE_3: tl.constexpr,
    Y_STRIDE_4: tl.constexpr,
    Y_STRIDE_5: tl.constexpr,
    Y_STRIDE_6: tl.constexpr,
    Y_STRIDE_7: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    block_base = program_id * BLOCK_SIZE
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    while block_base < n_elements:
        logical_index = block_base + lane
        mask = logical_index < n_elements
        remaining = logical_index
        x_offset = logical_index * 0
        y_offset = logical_index * 0

        coordinate = remaining % DIM_7
        remaining //= DIM_7
        x_offset += coordinate * X_STRIDE_7
        y_offset += coordinate * Y_STRIDE_7
        coordinate = remaining % DIM_6
        remaining //= DIM_6
        x_offset += coordinate * X_STRIDE_6
        y_offset += coordinate * Y_STRIDE_6
        coordinate = remaining % DIM_5
        remaining //= DIM_5
        x_offset += coordinate * X_STRIDE_5
        y_offset += coordinate * Y_STRIDE_5
        coordinate = remaining % DIM_4
        remaining //= DIM_4
        x_offset += coordinate * X_STRIDE_4
        y_offset += coordinate * Y_STRIDE_4
        coordinate = remaining % DIM_3
        remaining //= DIM_3
        x_offset += coordinate * X_STRIDE_3
        y_offset += coordinate * Y_STRIDE_3
        coordinate = remaining % DIM_2
        remaining //= DIM_2
        x_offset += coordinate * X_STRIDE_2
        y_offset += coordinate * Y_STRIDE_2
        coordinate = remaining % DIM_1
        remaining //= DIM_1
        x_offset += coordinate * X_STRIDE_1
        y_offset += coordinate * Y_STRIDE_1
        coordinate = remaining % DIM_0
        x_offset += coordinate * X_STRIDE_0
        y_offset += coordinate * Y_STRIDE_0

        channel = (logical_index // SPATIAL) % CHANNELS
        value = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        inv_variance = tl.load(
            inv_variance_ptr + channel, mask=mask, other=0.0
        ).to(tl.float32)
        scale = tl.load(scale_ptr + channel, mask=mask, other=0.0).to(
            tl.float32
        )
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        result = (value - mean) * inv_variance * scale + bias
        tl.store(y_ptr + y_offset, result.to(y_ptr.dtype.element_ty), mask=mask)
        block_base += BLOCK_SIZE * WORKER_COUNT
