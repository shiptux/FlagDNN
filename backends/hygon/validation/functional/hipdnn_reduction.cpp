/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/reduction.hpp"
#include "tensor_runner_support.hpp"

#include <memory>

namespace flagdnn::testing {

TestTensor
reduction_reference_input_tensor(const ReductionTestCase &test_case) {
  validate_reduction_case(test_case);
  return test_case.input;
}

std::unique_ptr<ReductionExecutable>
build_reduction_reference(const ReductionTestCase &test_case) {
  namespace support = hygon_functional::tensor;
  namespace hv = validation::hygon;
  validate_reduction_case(test_case);
  return std::make_unique<support::ReferenceExecutable>(
      hv::make_hipdnn_reduction_operation(test_case.mode, test_case.axis,
                                          test_case.keep_dimensions),
      support::reduction_reference_tensors(test_case));
}

} // namespace flagdnn::testing
