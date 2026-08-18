/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/benchmark/aclnn_batchnorm_inference_provider.hpp"

#include "common/normalization.hpp"
#include "validation/batchnorm_inference_validation.hpp"
#include "validation/functional/aclnn_batchnorm_inference.hpp"

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

flagdnn::testing::BatchnormInferenceTestCase to_test_case(
    const BenchmarkCase& specification) {
  const BenchmarkCase reference =
      flagdnn::validation::ascend::
          batchnorm_inference_reference_benchmark_case(specification);
  flagdnn::testing::BatchnormInferenceTestCase result;
  result.name = reference.name;
  result.x = to_test_tensor(reference.tensors[0]);
  result.mean = to_test_tensor(reference.tensors[1]);
  result.inv_variance = to_test_tensor(reference.tensors[2]);
  result.scale = to_test_tensor(reference.tensors[3]);
  result.bias = to_test_tensor(reference.tensors[4]);
  result.y = to_test_tensor(reference.tensors[5]);
  result.absolute_tolerance = reference.absolute_tolerance;
  result.relative_tolerance = reference.relative_tolerance;
  return result;
}

class AclnnBatchnormInferenceBenchmarkExecutable final
    : public BenchmarkExecutable {
 public:
  explicit AclnnBatchnormInferenceBenchmarkExecutable(
      const BenchmarkCase& specification)
      : implementation_(
            flagdnn::testing::build_batchnorm_inference_reference(
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

std::unique_ptr<BenchmarkExecutable> build_aclnn_batchnorm_inference(
    const BenchmarkCase& specification) {
  return std::make_unique<AclnnBatchnormInferenceBenchmarkExecutable>(
      specification);
}

}  // namespace flagdnn::benchmarking
