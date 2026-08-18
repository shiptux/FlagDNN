/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"
#include "pointwise_runner_support.hpp"

#include <memory>

namespace flagdnn::testing {

std::unique_ptr<PointwiseExecutable>
build_pointwise_reference(const PointwiseTestCase &test_case) {
  namespace support = hygon_functional::pointwise;
  namespace hv = validation::hygon;
  return std::make_unique<support::ReferenceExecutable>(
      hv::make_hipdnn_pointwise_operation(test_case.mode, test_case.attributes,
                                          test_case.alpha),
      support::reference_tensors(test_case.inputs, test_case.output));
}

} // namespace flagdnn::testing
