/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/benchmark/aclnn_layernorm_provider.hpp"

#include "common/normalization.hpp"
#include "validation/functional/aclnn_layernorm.hpp"
#include "validation/layernorm_validation.hpp"

#include <memory>
#include <span>

namespace flagdnn::benchmarking {
namespace {

flagdnn::testing::TestTensor to_test_tensor(const TensorSpec& tensor) {
  return {tensor.uid,
          tensor.data_type,
          tensor.dimensions,
          tensor.strides,
          tensor.binding_byte_offset};
}

flagdnn::testing::LayernormTestCase to_test_case(
    const BenchmarkCase& specification) {
  (void)flagdnn::validation::ascend::plan_layernorm(specification);
  flagdnn::testing::LayernormTestCase result;
  result.name = specification.name;
  result.x = to_test_tensor(specification.tensors[0]);
  result.scale = to_test_tensor(specification.tensors[1]);
  result.bias = to_test_tensor(specification.tensors[2]);
  result.y = to_test_tensor(specification.tensors[3]);
  result.mean = to_test_tensor(specification.tensors[4]);
  result.inv_variance = to_test_tensor(specification.tensors[5]);
  result.epsilon = specification.normalization.epsilon;
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  return result;
}

class AclnnLayernormBenchmarkExecutable final : public BenchmarkExecutable {
 public:
  explicit AclnnLayernormBenchmarkExecutable(
      const BenchmarkCase& specification)
      : implementation_(flagdnn::testing::build_layernorm_reference(
            to_test_case(specification))) {}

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
  std::unique_ptr<flagdnn::testing::NormalizationExecutable> implementation_;
};

}  // namespace

std::unique_ptr<BenchmarkExecutable> build_aclnn_layernorm(
    const BenchmarkCase& specification) {
  return std::make_unique<AclnnLayernormBenchmarkExecutable>(specification);
}

}  // namespace flagdnn::benchmarking
