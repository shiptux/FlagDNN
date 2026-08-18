/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_rmsnorm.hpp"

#include "validation/tensor_io.hpp"

namespace flagdnn::testing {

AclnnRmsnormPlan plan_aclnn_rmsnorm(const RmsnormTestCase& test_case) {
  AclnnRmsnormPlan result;
  result.operation = validation::ascend::plan_rmsnorm(test_case);
  result.gamma = result.operation.scale;
  while (result.gamma.dimensions.size() > 1U &&
         result.gamma.dimensions.front() == 1) {
    result.gamma.dimensions.erase(result.gamma.dimensions.begin());
    result.gamma.strides.erase(result.gamma.strides.begin());
  }
  result.normalized_output = result.operation.output;
  result.normalized_output.uid = 0;
  result.normalized_output.binding_byte_offset = 0;
  result.normalized_output_bytes =
      validation::ascend::tensor_io::encoded_byte_count(
          result.normalized_output);
  return result;
}

}  // namespace flagdnn::testing
