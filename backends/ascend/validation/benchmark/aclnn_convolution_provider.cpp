/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/benchmark/aclnn_convolution_provider.hpp"

#include "common/convolution.hpp"
#include "validation/functional/aclnn_convolution.hpp"

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

flagdnn::testing::ConvolutionTestCase to_test_case(
    const BenchmarkCase& specification) {
  if (specification.operation != Operation::kConvolutionFprop ||
      specification.output_count != 1U ||
      specification.tensors.size() != 3U) {
    throw std::invalid_argument(
        "ACLNN convolution fprop benchmark requires two inputs and one output");
  }
  flagdnn::testing::ConvolutionTestCase result;
  result.name = specification.name;
  result.direction = flagdnn::testing::ConvolutionDirection::kFprop;
  result.x = to_test_tensor(specification.tensors[0]);
  result.w = to_test_tensor(specification.tensors[1]);
  result.y = to_test_tensor(specification.tensors[2]);
  result.pre_padding = specification.convolution.pre_padding;
  result.post_padding = specification.convolution.post_padding;
  result.stride = specification.convolution.stride;
  result.dilation = specification.convolution.dilation;
  result.groups = specification.convolution.groups;
  result.mode =
      specification.convolution.mode == ConvolutionMode::kCrossCorrelation
          ? flagdnn::testing::ConvolutionMode::kCrossCorrelation
          : flagdnn::testing::ConvolutionMode::kConvolution;
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  result.autotune = false;
  flagdnn::testing::validate_convolution_case(result);
  return result;
}

class AclnnConvolutionFpropBenchmarkExecutable final
    : public BenchmarkExecutable {
 public:
  explicit AclnnConvolutionFpropBenchmarkExecutable(
      const BenchmarkCase& specification)
      : implementation_(flagdnn::testing::build_convolution_reference(
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
  std::unique_ptr<flagdnn::testing::ConvolutionExecutable> implementation_;
};

}  // namespace

AclnnConvolutionFpropBenchmarkPlan
plan_aclnn_convolution_fprop_benchmark(
    const BenchmarkCase& specification) {
  const flagdnn::testing::AclnnConvolutionFpropPlan plan =
      flagdnn::testing::plan_aclnn_convolution_fprop(
          to_test_case(specification));
  return {to_tensor_spec(plan.input),
          to_tensor_spec(plan.filter),
          to_tensor_spec(plan.output),
          to_tensor_spec(plan.convolution_input),
          plan.explicit_padding,
          plan.convolution_padding,
          plan.stride,
          plan.dilation,
          plan.groups};
}

std::unique_ptr<BenchmarkExecutable> build_aclnn_convolution_fprop(
    const BenchmarkCase& specification) {
  (void)plan_aclnn_convolution_fprop_benchmark(specification);
  return std::make_unique<AclnnConvolutionFpropBenchmarkExecutable>(
      specification);
}

}  // namespace flagdnn::benchmarking
