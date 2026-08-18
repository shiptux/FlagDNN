/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/layout.hpp"
#include "tensor_runner_support.hpp"

#include <memory>

namespace flagdnn::testing {

std::unique_ptr<LayoutExecutable>
build_layout_reference(const LayoutTestCase &test_case) {
  namespace support = hygon_functional::tensor;
  validate_layout_case(test_case);
  return std::make_unique<support::ReferenceExecutable>(
      support::layout_operation(test_case),
      support::layout_reference_tensors(test_case));
}

} // namespace flagdnn::testing
