/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/matmul.hpp"
#include "tensor_runner_support.hpp"

#include <memory>

namespace flagdnn::testing {

std::unique_ptr<MatmulExecutable>
build_matmul_reference(const MatmulTestCase &test_case) {
  namespace support = hygon_functional::tensor;
  validate_matmul_case(test_case);
  return std::make_unique<support::ReferenceExecutable>(
      support::matmul_operation(),
      support::matmul_reference_tensors(test_case));
}

} // namespace flagdnn::testing
