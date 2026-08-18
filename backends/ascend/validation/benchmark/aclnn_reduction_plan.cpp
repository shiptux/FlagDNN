/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/benchmark/aclnn_provider.hpp"

#include "validation/tensor_io.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace flagdnn::benchmarking {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;

AclnnReductionBenchmarkPlan plan_aclnn_reduction(
    const BenchmarkCase& specification) {
  if (specification.operation != Operation::kReduction ||
      specification.tensors.size() != 2 ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "ACLNN reduction benchmark requires one input and one output");
  }
  AclnnReductionBenchmarkPlan result{specification.tensors[0],
                                     specification.tensors[1],
                                     specification.reduction_mode,
                                     specification.reduction_axis,
                                     specification.keep_dimensions};
  if (result.input.dimensions.empty() ||
      result.input.dimensions.size() > 8 ||
      result.input.data_type != result.output.data_type ||
      (result.input.data_type != FLAGDNN_DATA_FLOAT32 &&
       result.input.data_type != FLAGDNN_DATA_FLOAT16 &&
       result.input.data_type != FLAGDNN_DATA_BFLOAT16)) {
    throw std::invalid_argument(
        "ACLNN reduction benchmark tensor metadata is invalid");
  }
  if (result.mode != FLAGDNN_REDUCTION_ADD &&
      result.mode != FLAGDNN_REDUCTION_AVG &&
      result.mode != FLAGDNN_REDUCTION_MUL) {
    throw std::invalid_argument(
        "ACLNN reduction benchmark mode is invalid");
  }
  if (result.axis < 0) {
    result.axis += static_cast<std::int64_t>(result.input.dimensions.size());
  }
  if (result.axis < 0 ||
      static_cast<std::size_t>(result.axis) >=
          result.input.dimensions.size()) {
    throw std::invalid_argument(
        "ACLNN reduction benchmark axis is out of range");
  }
  std::vector<std::int64_t> expected = result.input.dimensions;
  if (result.keep_dimensions) {
    expected[static_cast<std::size_t>(result.axis)] = 1;
  } else {
    expected.erase(expected.begin() + result.axis);
  }
  if (result.output.dimensions != expected) {
    throw std::invalid_argument(
        "ACLNN reduction benchmark output shape is invalid");
  }
  // The independent reference uses an aligned entrance even if a future
  // benchmark deliberately offsets the production binding.
  result.input.binding_byte_offset = 0;
  (void)tensor_io::encoded_byte_count(result.input);
  (void)tensor_io::encoded_byte_count(result.output);
  return result;
}

}  // namespace flagdnn::benchmarking
