/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/convolution.hpp"
#include "convolution_runner_support.hpp"

#include <memory>
#include <vector>

namespace flagdnn::testing {

std::unique_ptr<ConvolutionExecutable>
build_convolution_reference(const ConvolutionTestCase &test_case) {
  namespace support = hygon_functional::convolution;
  const std::vector<TestTensor> tensors = support::semantic_tensors(test_case);
  return std::make_unique<support::ReferenceExecutable>(
      support::reference_operation(test_case),
      support::reference_tensors(tensors));
}

} // namespace flagdnn::testing
