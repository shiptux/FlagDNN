/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_layernorm.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace flagdnn::testing {

AclnnLayernormPlan plan_aclnn_layernorm(
    const LayernormTestCase& test_case) {
  AclnnLayernormPlan result;
  result.operation = validation::ascend::plan_layernorm(test_case);
  std::int64_t product = 1;
  for (std::size_t axis = result.operation.input.dimensions.size();
       axis != 0;
       --axis) {
    const std::int64_t dimension =
        result.operation.input.dimensions[axis - 1];
    if (product > result.operation.normalized_elements / dimension) {
      throw std::invalid_argument(
          "ACLNN LayerNorm normalized shape is invalid");
    }
    product *= dimension;
    result.normalized_shape.insert(result.normalized_shape.begin(), dimension);
    if (product == result.operation.normalized_elements) {
      break;
    }
  }
  if (product != result.operation.normalized_elements ||
      result.normalized_shape.empty()) {
    throw std::invalid_argument("ACLNN LayerNorm normalized shape is invalid");
  }
  return result;
}

}  // namespace flagdnn::testing
