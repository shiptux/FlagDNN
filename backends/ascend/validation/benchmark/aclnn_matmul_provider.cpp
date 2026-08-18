/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/benchmark/aclnn_matmul_provider.hpp"

#include "common/matmul.hpp"
#include "validation/functional/aclnn_matmul.hpp"

#include <memory>
#include <span>
#include <stdexcept>

namespace flagdnn::benchmarking {
namespace {

flagdnn::testing::TestTensor to_test_tensor(const TensorSpec& tensor) {
  return {tensor.uid,
          tensor.data_type,
          tensor.dimensions,
          tensor.strides,
          tensor.binding_byte_offset};
}

TensorSpec to_tensor_spec(const flagdnn::testing::TestTensor& tensor) {
  return {tensor.uid,
          tensor.data_type,
          tensor.dimensions,
          tensor.strides,
          tensor.binding_byte_offset};
}

flagdnn::testing::MatmulTestCase to_test_case(
    const BenchmarkCase& specification) {
  if (specification.operation != Operation::kMatmul ||
      specification.output_count != 1 ||
      specification.tensors.size() != 3) {
    throw std::invalid_argument(
        "ACLNN MatMul benchmark requires two inputs and one output");
  }
  flagdnn::testing::MatmulTestCase result;
  result.name = specification.name;
  result.a = to_test_tensor(specification.tensors[0]);
  result.b = to_test_tensor(specification.tensors[1]);
  result.output = to_test_tensor(specification.tensors[2]);
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  result.autotune = false;
  flagdnn::testing::validate_matmul_case(result);
  return result;
}

class AclnnMatmulBenchmarkExecutable final : public BenchmarkExecutable {
 public:
  explicit AclnnMatmulBenchmarkExecutable(
      const BenchmarkCase& specification)
      : implementation_(flagdnn::testing::build_matmul_reference(
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
  std::unique_ptr<flagdnn::testing::MatmulExecutable> implementation_;
};

}  // namespace

AclnnMatmulBenchmarkPlan plan_aclnn_matmul_benchmark(
    const BenchmarkCase& specification) {
  const flagdnn::testing::AclnnMatmulPlan plan =
      flagdnn::testing::plan_aclnn_matmul(to_test_case(specification));
  return {to_tensor_spec(plan.a),
          to_tensor_spec(plan.b),
          to_tensor_spec(plan.output),
          plan.cube_math_type};
}

std::unique_ptr<BenchmarkExecutable> build_aclnn_matmul(
    const BenchmarkCase& specification) {
  (void)plan_aclnn_matmul_benchmark(specification);
  return std::make_unique<AclnnMatmulBenchmarkExecutable>(specification);
}

}  // namespace flagdnn::benchmarking
