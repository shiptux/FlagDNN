# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hygon-specialized identity kernels.

The entry points deliberately keep the common unary ABI so the graph planner,
libtriton_jit argument packing, and installed-resource contract stay identical
across backends.  Only the identity operation is registered to this module.
"""

import triton
import triton.language as tl


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
    TILES_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy one dense tile without an arithmetic/conversion round trip."""

    # n_elements has an i32 ABI and is bounded accordingly by the graph
    # compiler. Keeping the dense offset arithmetic in i32 avoids an
    # unnecessary 64-bit multiply on gfx936. Multiple statically-unrolled
    # tiles give each workgroup enough independent memory operations to
    # amortize scheduling overhead for large bandwidth-bound copies.
    program_base = tl.program_id(0) * BLOCK_SIZE * TILES_PER_PROGRAM
    for tile_index in tl.static_range(TILES_PER_PROGRAM):
        offsets = (
            program_base + tile_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        active = offsets < n_elements
        value = tl.load(in_ptr + offsets, mask=active, other=0.0)
        tl.store(out_ptr + offsets, value, mask=active)


@triton.jit
def identity_packed_contiguous_kernel(
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
    PACK_FACTOR: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy aligned dense identity tensors as raw 64-bit words.

    The compiler selects this entry point only when PACK_FACTOR logical
    elements occupy exactly one uint64 and n_elements is divisible by it.
    Reinterpreting the pointers preserves every source bit while reducing the
    number of global-memory instructions for FP16/BF16 and FP32 identity.
    """

    packed_elements = n_elements // PACK_FACTOR
    packed_in_ptr = in_ptr.to(tl.pointer_type(tl.uint64))
    packed_out_ptr = out_ptr.to(tl.pointer_type(tl.uint64))
    program_base = tl.program_id(0) * BLOCK_SIZE * TILES_PER_PROGRAM
    for tile_index in tl.static_range(TILES_PER_PROGRAM):
        offsets = (
            program_base + tile_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        offsets = tl.max_contiguous(offsets, BLOCK_SIZE)
        active = offsets < packed_elements
        value = tl.load(packed_in_ptr + offsets, mask=active, other=0)
        tl.store(packed_out_ptr + offsets, value, mask=active)


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
    STRIDED: tl.constexpr,
    OPERATION: tl.constexpr,
    negative_slope: tl.constexpr,
    lower_clip: tl.constexpr,
    upper_clip: tl.constexpr,
    HAS_UPPER_CLIP: tl.constexpr,
    SWISH_BETA: tl.constexpr,
    ELU_ALPHA: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Identity fallback for every rank/stride accepted by the graph ABI."""

    program_base = (
        tl.program_id(0).to(tl.int64) * BLOCK_SIZE * TILES_PER_PROGRAM
    )
    for tile_index in tl.static_range(TILES_PER_PROGRAM):
        offsets = (
            program_base + tile_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        active = offsets < n_elements
        remaining = offsets
        input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        coordinate = remaining % DIM_7
        remaining //= DIM_7
        input_offsets += coordinate * INPUT_STRIDE_7
        output_offsets += coordinate * OUTPUT_STRIDE_7
        coordinate = remaining % DIM_6
        remaining //= DIM_6
        input_offsets += coordinate * INPUT_STRIDE_6
        output_offsets += coordinate * OUTPUT_STRIDE_6
        coordinate = remaining % DIM_5
        remaining //= DIM_5
        input_offsets += coordinate * INPUT_STRIDE_5
        output_offsets += coordinate * OUTPUT_STRIDE_5
        coordinate = remaining % DIM_4
        remaining //= DIM_4
        input_offsets += coordinate * INPUT_STRIDE_4
        output_offsets += coordinate * OUTPUT_STRIDE_4
        coordinate = remaining % DIM_3
        remaining //= DIM_3
        input_offsets += coordinate * INPUT_STRIDE_3
        output_offsets += coordinate * OUTPUT_STRIDE_3
        coordinate = remaining % DIM_2
        remaining //= DIM_2
        input_offsets += coordinate * INPUT_STRIDE_2
        output_offsets += coordinate * OUTPUT_STRIDE_2
        coordinate = remaining % DIM_1
        remaining //= DIM_1
        input_offsets += coordinate * INPUT_STRIDE_1
        output_offsets += coordinate * OUTPUT_STRIDE_1
        coordinate = remaining % DIM_0
        input_offsets += coordinate * INPUT_STRIDE_0
        output_offsets += coordinate * OUTPUT_STRIDE_0

        value = tl.load(in_ptr + input_offsets, mask=active, other=0.0)
        tl.store(out_ptr + output_offsets, value, mask=active)
