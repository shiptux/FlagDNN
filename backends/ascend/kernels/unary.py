# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend-owned persistent unary kernels for the LTJ NPU contract."""

import triton
import triton.language as tl
from triton.language.extra.cann import libdevice as cann_libdevice


POINTWISE_RELU = tl.constexpr(2)
POINTWISE_SQRT = tl.constexpr(3)
POINTWISE_ERF = tl.constexpr(4)
POINTWISE_IDENTITY = tl.constexpr(5)
POINTWISE_EXP = tl.constexpr(6)
POINTWISE_LOG = tl.constexpr(7)
POINTWISE_NEG = tl.constexpr(8)
POINTWISE_ABS = tl.constexpr(9)
POINTWISE_CEIL = tl.constexpr(10)
POINTWISE_COS = tl.constexpr(11)
POINTWISE_FLOOR = tl.constexpr(12)
POINTWISE_RSQRT = tl.constexpr(13)
POINTWISE_SIN = tl.constexpr(14)
POINTWISE_TAN = tl.constexpr(15)
POINTWISE_RECIPROCAL = tl.constexpr(16)
POINTWISE_LOGICAL_NOT = tl.constexpr(24)
POINTWISE_SIGMOID = tl.constexpr(33)
POINTWISE_TANH = tl.constexpr(34)
POINTWISE_ELU = tl.constexpr(35)
POINTWISE_GELU = tl.constexpr(36)
POINTWISE_SOFTPLUS = tl.constexpr(37)
POINTWISE_SWISH = tl.constexpr(38)
POINTWISE_GELU_APPROX_TANH = tl.constexpr(39)


@triton.jit
def _apply_unary_operation(
    value,
    OPERATION: tl.constexpr,
    negative_slope: tl.constexpr,
    lower_clip: tl.constexpr,
    upper_clip: tl.constexpr,
    HAS_UPPER_CLIP: tl.constexpr,
    SWISH_BETA: tl.constexpr,
    ELU_ALPHA: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
):
    value_f32 = value.to(tl.float32)
    if OPERATION == POINTWISE_LOGICAL_NOT:
        result = (value == 0).to(tl.int8)
    elif OPERATION == POINTWISE_RELU:
        result = tl.where(
            value_f32 < lower_clip,
            lower_clip + negative_slope * (value_f32 - lower_clip),
            value_f32,
        )
        if HAS_UPPER_CLIP:
            result = tl.minimum(result, upper_clip)
    elif OPERATION == POINTWISE_SQRT:
        result = tl.sqrt(value_f32)
    elif OPERATION == POINTWISE_ERF:
        result = tl.erf(value_f32)
    elif OPERATION == POINTWISE_IDENTITY:
        result = value
    elif OPERATION == POINTWISE_EXP:
        result = tl.exp(value_f32)
    elif OPERATION == POINTWISE_LOG:
        result = tl.log(value_f32)
    elif OPERATION == POINTWISE_NEG:
        result = -value_f32
    elif OPERATION == POINTWISE_ABS:
        result = tl.abs(value_f32)
    elif OPERATION == POINTWISE_CEIL:
        result = tl.ceil(value_f32)
    elif OPERATION == POINTWISE_COS:
        result = tl.cos(value_f32)
    elif OPERATION == POINTWISE_FLOOR:
        result = tl.floor(value_f32)
    elif OPERATION == POINTWISE_RSQRT:
        result = tl.rsqrt(value_f32)
    elif OPERATION == POINTWISE_SIN:
        result = tl.sin(value_f32)
    elif OPERATION == POINTWISE_TAN:
        result = cann_libdevice.tan(value_f32)
    elif OPERATION == POINTWISE_RECIPROCAL:
        result = 1.0 / value_f32
    elif OPERATION == POINTWISE_SIGMOID:
        result = tl.sigmoid(value_f32)
    elif OPERATION == POINTWISE_TANH:
        result = 2.0 * tl.sigmoid(2.0 * value_f32) - 1.0
    elif OPERATION == POINTWISE_ELU:
        result = tl.where(
            value_f32 > 0.0,
            value_f32,
            ELU_ALPHA * (tl.exp(value_f32) - 1.0),
        )
    elif OPERATION == POINTWISE_GELU:
        result = 0.5 * value_f32 * (1.0 + tl.erf(value_f32 * 0.7071067811865476))
    elif OPERATION == POINTWISE_SOFTPLUS:
        scaled = SOFTPLUS_BETA * value_f32
        result = (
            tl.maximum(scaled, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(scaled)))
        ) / SOFTPLUS_BETA
    elif OPERATION == POINTWISE_SWISH:
        result = value_f32 * tl.sigmoid(SWISH_BETA * value_f32)
    elif OPERATION == POINTWISE_GELU_APPROX_TANH:
        cubic = value_f32 * value_f32 * value_f32
        approximate_argument = 0.7978845608028654 * (
            value_f32 + 0.044715 * cubic
        )
        approximate_tanh = (
            2.0 * tl.sigmoid(2.0 * approximate_argument) - 1.0
        )
        result = 0.5 * value_f32 * (1.0 + approximate_tanh)
    return result


@triton.jit
def unary_pointwise_contiguous_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    OPERATION: tl.constexpr,
    negative_slope: tl.constexpr,
    lower_clip: tl.constexpr,
    upper_clip: tl.constexpr,
    HAS_UPPER_CLIP: tl.constexpr,
    SWISH_BETA: tl.constexpr,
    ELU_ALPHA: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    start = program_id * BLOCK_SIZE
    while start < n_elements:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        value = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        result = _apply_unary_operation(
            value,
            OPERATION,
            negative_slope,
            lower_clip,
            upper_clip,
            HAS_UPPER_CLIP,
            SWISH_BETA,
            ELU_ALPHA,
            SOFTPLUS_BETA,
        )
        tl.store(
            out_ptr + offsets,
            result.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
        start += WORKER_COUNT * BLOCK_SIZE


@triton.jit
def unary_pointwise_strided_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    DIM_0: tl.constexpr,
    DIM_1: tl.constexpr,
    DIM_2: tl.constexpr,
    DIM_3: tl.constexpr,
    DIM_4: tl.constexpr,
    DIM_5: tl.constexpr,
    DIM_6: tl.constexpr,
    DIM_7: tl.constexpr,
    INPUT_STRIDE_0: tl.constexpr,
    INPUT_STRIDE_1: tl.constexpr,
    INPUT_STRIDE_2: tl.constexpr,
    INPUT_STRIDE_3: tl.constexpr,
    INPUT_STRIDE_4: tl.constexpr,
    INPUT_STRIDE_5: tl.constexpr,
    INPUT_STRIDE_6: tl.constexpr,
    INPUT_STRIDE_7: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    OUTPUT_STRIDE_2: tl.constexpr,
    OUTPUT_STRIDE_3: tl.constexpr,
    OUTPUT_STRIDE_4: tl.constexpr,
    OUTPUT_STRIDE_5: tl.constexpr,
    OUTPUT_STRIDE_6: tl.constexpr,
    OUTPUT_STRIDE_7: tl.constexpr,
    OPERATION: tl.constexpr,
    negative_slope: tl.constexpr,
    lower_clip: tl.constexpr,
    upper_clip: tl.constexpr,
    HAS_UPPER_CLIP: tl.constexpr,
    SWISH_BETA: tl.constexpr,
    ELU_ALPHA: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    start = program_id * BLOCK_SIZE
    while start < n_elements:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        remaining = offsets
        input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        coordinate = remaining % DIM_7
        remaining = remaining // DIM_7
        input_offsets += coordinate * INPUT_STRIDE_7
        output_offsets += coordinate * OUTPUT_STRIDE_7
        coordinate = remaining % DIM_6
        remaining = remaining // DIM_6
        input_offsets += coordinate * INPUT_STRIDE_6
        output_offsets += coordinate * OUTPUT_STRIDE_6
        coordinate = remaining % DIM_5
        remaining = remaining // DIM_5
        input_offsets += coordinate * INPUT_STRIDE_5
        output_offsets += coordinate * OUTPUT_STRIDE_5
        coordinate = remaining % DIM_4
        remaining = remaining // DIM_4
        input_offsets += coordinate * INPUT_STRIDE_4
        output_offsets += coordinate * OUTPUT_STRIDE_4
        coordinate = remaining % DIM_3
        remaining = remaining // DIM_3
        input_offsets += coordinate * INPUT_STRIDE_3
        output_offsets += coordinate * OUTPUT_STRIDE_3
        coordinate = remaining % DIM_2
        remaining = remaining // DIM_2
        input_offsets += coordinate * INPUT_STRIDE_2
        output_offsets += coordinate * OUTPUT_STRIDE_2
        coordinate = remaining % DIM_1
        remaining = remaining // DIM_1
        input_offsets += coordinate * INPUT_STRIDE_1
        output_offsets += coordinate * OUTPUT_STRIDE_1
        coordinate = remaining % DIM_0
        input_offsets += coordinate * INPUT_STRIDE_0
        output_offsets += coordinate * OUTPUT_STRIDE_0

        value = tl.load(in_ptr + input_offsets, mask=mask, other=0.0)
        result = _apply_unary_operation(
            value,
            OPERATION,
            negative_slope,
            lower_clip,
            upper_clip,
            HAS_UPPER_CLIP,
            SWISH_BETA,
            ELU_ALPHA,
            SOFTPLUS_BETA,
        )
        tl.store(
            out_ptr + output_offsets,
            result.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
        start += WORKER_COUNT * BLOCK_SIZE
