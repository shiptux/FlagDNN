/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/add.hpp"
#include "pointwise_runner_support.hpp"

#include <array>
#include <memory>

namespace flagdnn::testing {

std::unique_ptr<AddExecutable>
build_add_reference(const AddTestCase &test_case) {
  namespace support = hygon_functional::pointwise;
  namespace hv = validation::hygon;
  return std::make_unique<support::ReferenceExecutable>(
      support::fixed_operation(hv::HipdnnPointwiseKind::kAdd, test_case.alpha),
      support::reference_tensors(
          std::array<TestTensor, 2>{test_case.left, test_case.right},
          test_case.output));
}

} // namespace flagdnn::testing
