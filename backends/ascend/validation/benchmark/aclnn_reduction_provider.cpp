/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/benchmark/aclnn_provider.hpp"

#include "common/reduction.hpp"
#include "validation/functional/aclnn_reduction.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

namespace flagdnn::benchmarking {
namespace {

flagdnn::testing::TestTensor to_test_tensor(const TensorSpec& tensor) {
  return {tensor.uid,
          tensor.data_type,
          tensor.dimensions,
          tensor.strides,
          tensor.binding_byte_offset};
}

flagdnn::testing::ReductionTestCase to_test_case(
    const BenchmarkCase& specification,
    const AclnnReductionBenchmarkPlan& plan) {
  flagdnn::testing::ReductionTestCase result;
  result.name = specification.name;
  result.input = to_test_tensor(plan.input);
  result.output = to_test_tensor(plan.output);
  result.mode = plan.mode;
  result.axis = static_cast<std::int32_t>(plan.axis);
  result.keep_dimensions = plan.keep_dimensions;
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  flagdnn::testing::validate_reduction_case(result);
  return result;
}

class AclnnReductionBenchmarkExecutable final : public BenchmarkExecutable {
 public:
  AclnnReductionBenchmarkExecutable(
      const BenchmarkCase& specification,
      const AclnnReductionBenchmarkPlan& plan)
      : implementation_(flagdnn::testing::build_reduction_reference(
            to_test_case(specification, plan))) {}

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    implementation_->prepare(bindings, stream);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return implementation_->workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    implementation_->execute(bindings, workspace, workspace_size, stream);
  }

 private:
  std::unique_ptr<flagdnn::testing::ReductionExecutable> implementation_;
};

}  // namespace

std::unique_ptr<BenchmarkExecutable> build_aclnn_reduction(
    const BenchmarkCase& specification) {
  return std::make_unique<AclnnReductionBenchmarkExecutable>(
      specification, plan_aclnn_reduction(specification));
}

}  // namespace flagdnn::benchmarking
