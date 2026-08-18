/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_matmul.hpp"

#include "validation/tensor_io.hpp"

namespace flagdnn::testing {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;

AclnnMatmulPlan plan_aclnn_matmul(const MatmulTestCase& test_case) {
  validate_matmul_case(test_case);
  AclnnMatmulPlan result;
  result.a = test_case.a;
  result.b = test_case.b;
  result.output = test_case.output;
  // The reference owns independent aligned allocations.  Do not inherit a
  // deliberately offset FlagDNN binding entrance into aclCreateTensor.
  result.a.binding_byte_offset = 0;
  result.b.binding_byte_offset = 0;
  result.output.binding_byte_offset = 0;
  result.cube_math_type = 0;  // ACLNN KEEP_DTYPE; never permit implicit TF32.
  (void)tensor_io::encoded_byte_count(result.a);
  (void)tensor_io::encoded_byte_count(result.b);
  (void)tensor_io::encoded_byte_count(result.output);
  return result;
}

}  // namespace flagdnn::testing
