/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_reduction.hpp"

#include "validation/tensor_io.hpp"

#include <cstdint>

namespace flagdnn::testing {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;

AclnnReductionPlan plan_aclnn_reduction(
    const ReductionTestCase& test_case) {
  validate_reduction_case(test_case);
  AclnnReductionPlan result;
  result.input = test_case.input;
  // ACLNN's independent allocation starts at an aligned address even when
  // the FlagDNN case intentionally tests an unaligned binding entrance.
  result.input.binding_byte_offset = 0;
  result.output = test_case.output;
  result.mode = test_case.mode;
  result.axis = test_case.axis;
  if (result.axis < 0) {
    result.axis += static_cast<std::int64_t>(result.input.dimensions.size());
  }
  result.keep_dimensions = test_case.keep_dimensions;
  (void)tensor_io::encoded_byte_count(result.input);
  (void)tensor_io::encoded_byte_count(result.output);
  return result;
}

TestTensor reduction_reference_input_tensor(
    const ReductionTestCase& test_case) {
  return plan_aclnn_reduction(test_case).input;
}

}  // namespace flagdnn::testing
