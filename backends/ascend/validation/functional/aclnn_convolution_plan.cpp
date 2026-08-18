/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_convolution.hpp"

#include "validation/tensor_io.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;

std::vector<std::int64_t> contiguous_strides(
    const std::vector<std::int64_t>& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0U; --axis) {
    const std::int64_t dimension = dimensions[axis - 1U];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::overflow_error(
          "ACLNN convolution padded tensor shape overflows int64");
    }
    result[axis - 1U] = stride;
    stride *= dimension;
  }
  return result;
}

}  // namespace

AclnnConvolutionFpropPlan plan_aclnn_convolution_fprop(
    const ConvolutionTestCase& test_case) {
  validate_convolution_case(test_case);
  if (test_case.direction != ConvolutionDirection::kFprop) {
    throw std::invalid_argument(
        "ACLNN convolution fprop plan requires the fprop direction");
  }
  if (test_case.mode != ConvolutionMode::kCrossCorrelation) {
    throw std::invalid_argument(
        "ACLNN convolution fprop reference supports cross-correlation only");
  }

  AclnnConvolutionFpropPlan result;
  result.input = test_case.x;
  result.filter = test_case.w;
  result.output = test_case.y;
  result.input.binding_byte_offset = 0U;
  result.filter.binding_byte_offset = 0U;
  result.output.binding_byte_offset = 0U;
  result.convolution_input = result.input;
  result.stride = test_case.stride;
  result.dilation = test_case.dilation;
  result.output_padding.assign(test_case.stride.size(), 0);
  result.groups = test_case.groups;
  result.cube_math_type = 0;
  result.transposed = false;

  const std::size_t spatial_rank = test_case.stride.size();
  const bool asymmetric =
      test_case.pre_padding != test_case.post_padding;
  if (!asymmetric) {
    result.convolution_padding = test_case.pre_padding;
  } else if (spatial_rank < 3U) {
    result.convolution_padding.reserve(spatial_rank * 2U);
    for (std::size_t axis = 0U; axis < spatial_rank; ++axis) {
      result.convolution_padding.push_back(test_case.pre_padding[axis]);
      result.convolution_padding.push_back(test_case.post_padding[axis]);
    }
  } else {
    result.convolution_padding.assign(spatial_rank, 0);
    result.explicit_padding.reserve(spatial_rank * 2U);
    for (std::size_t axis = spatial_rank; axis != 0U; --axis) {
      result.explicit_padding.push_back(test_case.pre_padding[axis - 1U]);
      result.explicit_padding.push_back(test_case.post_padding[axis - 1U]);
    }
    for (std::size_t axis = 0U; axis < spatial_rank; ++axis) {
      const std::int64_t before = test_case.pre_padding[axis];
      const std::int64_t after = test_case.post_padding[axis];
      std::int64_t& dimension =
          result.convolution_input.dimensions[axis + 2U];
      if (before > std::numeric_limits<std::int64_t>::max() - dimension ||
          after > std::numeric_limits<std::int64_t>::max() -
                      dimension - before) {
        throw std::overflow_error(
            "ACLNN convolution explicit padding shape overflows int64");
      }
      dimension += before + after;
    }
    result.convolution_input.strides =
        contiguous_strides(result.convolution_input.dimensions);
  }

  (void)tensor_io::encoded_byte_count(result.input);
  (void)tensor_io::encoded_byte_count(result.filter);
  (void)tensor_io::encoded_byte_count(result.output);
  (void)tensor_io::encoded_byte_count(result.convolution_input);
  return result;
}

}  // namespace flagdnn::testing
