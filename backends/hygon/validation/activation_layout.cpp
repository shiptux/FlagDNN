/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "activation_layout.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace flagdnn::validation::hygon {
namespace {

bool has_compact_physical_mapping(const ReferenceTensor &tensor) {
  if (tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    return false;
  }

  std::vector<std::pair<std::int64_t, std::int64_t>> physical_axes;
  physical_axes.reserve(tensor.dimensions.size());
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    const std::int64_t dimension = tensor.dimensions[axis];
    const std::int64_t stride = tensor.strides[axis];
    if (dimension <= 0 || stride <= 0 ||
        dimension > std::numeric_limits<int>::max() ||
        stride > std::numeric_limits<int>::max()) {
      return false;
    }
    if (dimension > 1) {
      physical_axes.emplace_back(stride, dimension);
    }
  }
  std::sort(physical_axes.begin(), physical_axes.end());

  std::int64_t expected_stride = 1;
  for (const auto &[stride, dimension] : physical_axes) {
    if (stride != expected_stride ||
        expected_stride >
            std::numeric_limits<std::int64_t>::max() / dimension) {
      return false;
    }
    expected_stride *= dimension;
  }
  return true;
}

bool has_same_physical_mapping(const ReferenceTensor &left,
                               const ReferenceTensor &right) {
  if (left.dimensions != right.dimensions ||
      left.strides.size() != right.strides.size()) {
    return false;
  }
  for (std::size_t axis = 0; axis < left.dimensions.size(); ++axis) {
    if (left.dimensions[axis] > 1 &&
        left.strides[axis] != right.strides[axis]) {
      return false;
    }
  }
  return true;
}

} // namespace

std::vector<ReferenceTensor>
hipdnn_activation_descriptor_tensors(std::span<const ReferenceTensor> tensors) {
  std::vector<ReferenceTensor> result(tensors.begin(), tensors.end());
  if (tensors.size() < 2) {
    return result;
  }

  const ReferenceTensor &first = tensors.front();
  if (first.binding_byte_offset != 0 || !has_compact_physical_mapping(first)) {
    return result;
  }
  for (const ReferenceTensor &tensor : tensors.subspan(1)) {
    if (tensor.data_type != first.data_type ||
        tensor.binding_byte_offset != 0 ||
        !has_compact_physical_mapping(tensor) ||
        !has_same_physical_mapping(first, tensor)) {
      return result;
    }
  }

  std::int64_t elements = 1;
  std::int64_t channels = 1;
  for (const std::int64_t dimension : first.dimensions) {
    if (dimension <= 0 ||
        elements > std::numeric_limits<std::int64_t>::max() / dimension) {
      return result;
    }
    elements *= dimension;
    if (dimension > 1) {
      channels = dimension;
    }
  }
  if (channels <= 0 || elements % channels != 0) {
    return result;
  }

  const std::int64_t batches = elements / channels;
  if (batches > std::numeric_limits<int>::max() ||
      channels > std::numeric_limits<int>::max()) {
    return result;
  }

  const std::vector<std::int64_t> dimensions = {batches, channels, 1, 1};
  const std::vector<std::int64_t> strides = {channels, 1, 1, 1};
  for (ReferenceTensor &tensor : result) {
    tensor.dimensions = dimensions;
    tensor.strides = strides;
  }
  return result;
}

} // namespace flagdnn::validation::hygon
