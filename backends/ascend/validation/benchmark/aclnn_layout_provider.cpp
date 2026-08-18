/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/benchmark/aclnn_layout_provider.hpp"

#include "common/layout.hpp"
#include "validation/functional/aclnn_layout.hpp"

#include <memory>
#include <span>
#include <stdexcept>
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

flagdnn::testing::LayoutTestCase to_layout_case(
    const BenchmarkCase& specification) {
  if (specification.output_count != 1 ||
      specification.tensors.size() != 2) {
    throw std::invalid_argument(
        "ACLNN layout benchmark requires one input and one output");
  }
  flagdnn::testing::LayoutTestCase result;
  result.name = specification.name;
  result.input = to_test_tensor(specification.tensors[0]);
  result.output = to_test_tensor(specification.tensors[1]);
  switch (specification.operation) {
    case Operation::kReshape:
      if (!specification.reshape.logical ||
          specification.reshape.dimensions != result.output.dimensions ||
          specification.reshape.strides != result.output.strides) {
        throw std::invalid_argument(
            "ACLNN layout benchmark requires an exact logical reshape");
      }
      result.operation = flagdnn::testing::LayoutOperation::kReshape;
      break;
    case Operation::kTranspose:
      result.operation = flagdnn::testing::LayoutOperation::kTranspose;
      result.permutation = specification.transpose.permutation;
      break;
    case Operation::kSlice:
      result.operation = flagdnn::testing::LayoutOperation::kSlice;
      result.slices = specification.slice.slices;
      result.slice_strides = specification.slice.strides;
      break;
    default:
      throw std::invalid_argument(
          "ACLNN layout benchmark operation is unsupported");
  }
  flagdnn::testing::validate_layout_case(result);
  return result;
}

class AclnnLayoutBenchmarkExecutable final : public BenchmarkExecutable {
 public:
  explicit AclnnLayoutBenchmarkExecutable(
      const BenchmarkCase& specification)
      : executable_(flagdnn::testing::build_layout_reference(
            to_layout_case(specification))) {}

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    try {
      executable_->prepare(bindings, stream);
    } catch (const flagdnn::testing::AclnnLayoutUnsupportedError& error) {
      throw AclnnLayoutBenchmarkUnsupportedError(error.status(),
                                                 error.what());
    }
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return executable_->workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    executable_->execute(bindings, workspace, workspace_size, stream);
  }

 private:
  std::unique_ptr<flagdnn::testing::LayoutExecutable> executable_;
};

}  // namespace

AclnnLayoutBenchmarkPlan plan_aclnn_layout_benchmark(
    const BenchmarkCase& specification) {
  (void)flagdnn::testing::plan_aclnn_layout(
      to_layout_case(specification));
  return {specification.tensors[0], specification.tensors[1]};
}

ProviderCapability AclnnLayoutProvider::capability(
    const BenchmarkCase& specification) const {
  if (specification.operation != Operation::kReshape &&
      specification.operation != Operation::kTranspose &&
      specification.operation != Operation::kSlice) {
    return ProviderCapability::unsupported(
        "no exact ACLNN layout mapping for this operation");
  }
  (void)plan_aclnn_layout_benchmark(specification);
  return {};
}

std::unique_ptr<BenchmarkExecutable> AclnnLayoutProvider::build(
    const BenchmarkCase& specification) {
  (void)plan_aclnn_layout_benchmark(specification);
  return std::make_unique<AclnnLayoutBenchmarkExecutable>(specification);
}

}  // namespace flagdnn::benchmarking
