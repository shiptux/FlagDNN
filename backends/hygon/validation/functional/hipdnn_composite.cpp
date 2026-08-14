/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/composite.hpp"
#include "convolution_runner_support.hpp"
#include "pointwise_runner_support.hpp"

#include <array>
#include <memory>
#include <vector>

namespace flagdnn::testing {

std::unique_ptr<CompositeExecutable>
build_add_square_reference(const AddSquareTestCase &test_case) {
  namespace support = hygon_functional::pointwise;
  namespace hv = validation::hygon;
  return std::make_unique<support::ReferenceExecutable>(
      support::fixed_operation(hv::HipdnnPointwiseKind::kAddSquare),
      support::reference_tensors(
          std::array<TestTensor, 2>{test_case.left, test_case.right},
          test_case.output));
}

std::unique_ptr<CompositeExecutable>
build_conv_bias_relu_reference(const ConvBiasReluTestCase &test_case) {
  namespace support = hygon_functional::convolution;
  const std::vector<TestTensor> tensors = support::semantic_tensors(test_case);
  return std::make_unique<support::ReferenceExecutable>(
      support::reference_operation(test_case),
      support::reference_tensors(tensors));
}

} // namespace flagdnn::testing
