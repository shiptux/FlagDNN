/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/benchmark/aclnn_provider.hpp"

#include "validation/aclnn_unary_runtime.hpp"
#include "validation/tensor_io.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

namespace acl = flagdnn::validation::ascend;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;

std::vector<std::int64_t> checked_contiguous_strides(
    const std::vector<std::int64_t>& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    result[axis - 1] = stride;
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::overflow_error(
          "ACLNN unary benchmark contiguous stride overflows int64");
    }
    stride *= dimension;
  }
  return result;
}

const flagdnnBinding_t& find_binding(
    std::span<const flagdnnBinding_t> bindings,
    std::int64_t uid) {
  const flagdnnBinding_t* result = nullptr;
  for (const flagdnnBinding_t& binding : bindings) {
    if (binding.uid != uid) {
      continue;
    }
    if (result != nullptr) {
      throw std::invalid_argument(
          "ACLNN unary benchmark binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument(
        "ACLNN unary benchmark binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument(
        "ACLNN unary benchmark binding UID " + std::to_string(uid) +
        " is not 32-byte aligned");
  }
}

acl::AclnnUnaryTensor runtime_tensor(const TensorSpec& tensor) {
  return {tensor.data_type, tensor.dimensions, tensor.strides};
}

class AclnnUnaryBenchmarkExecutable final : public BenchmarkExecutable {
 public:
  explicit AclnnUnaryBenchmarkExecutable(
      AclnnUnaryPointwiseBenchmarkPlan plan)
      : plan_(std::move(plan)), runtime_(plan_.mode, plan_.attributes) {}

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    const flagdnnBinding_t& input = find_binding(bindings, plan_.input.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    require_aligned(input.device_pointer, input.uid);
    require_aligned(output.device_pointer, output.uid);
    try {
      runtime_.prepare(runtime_tensor(plan_.input),
                       input.device_pointer,
                       runtime_tensor(plan_.output),
                       output.device_pointer,
                       stream);
    } catch (const acl::AclnnUnaryUnsupportedError& error) {
      throw AclnnBenchmarkUnsupportedError(error.status(), error.what());
    }
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return runtime_.workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    const flagdnnBinding_t& input = find_binding(bindings, plan_.input.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    runtime_.execute(input.device_pointer,
                     output.device_pointer,
                     workspace,
                     workspace_size,
                     stream);
  }

 private:
  AclnnUnaryPointwiseBenchmarkPlan plan_;
  acl::AclnnUnaryRuntime runtime_;
};

}  // namespace

bool is_aclnn_unary_benchmark(
    const BenchmarkCase& specification) noexcept {
  return specification.operation == Operation::kRelu ||
         (specification.operation == Operation::kPointwise &&
          acl::is_aclnn_unary_mode(specification.pointwise_mode));
}

AclnnUnaryPointwiseBenchmarkPlan plan_aclnn_unary_pointwise(
    const BenchmarkCase& specification) {
  if (!is_aclnn_unary_benchmark(specification) ||
      specification.tensors.size() != 2 ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "ACLNN unary provider requires one input and one output");
  }
  const flagdnnPointwiseMode_t mode =
      specification.operation == Operation::kRelu
          ? FLAGDNN_POINTWISE_RELU_FWD
          : specification.pointwise_mode;
  AclnnUnaryPointwiseBenchmarkPlan result{
      specification.tensors[0],
      specification.tensors[1],
      mode,
      specification.pointwise_attributes,
      false};
  if (result.input.data_type != result.output.data_type ||
      result.input.dimensions != result.output.dimensions) {
    throw std::invalid_argument(
        "ACLNN unary benchmark input and output metadata do not match");
  }
  const std::vector<std::int64_t> contiguous =
      checked_contiguous_strides(result.output.dimensions);
  if (result.output.strides != contiguous) {
    result.output.strides = contiguous;
    result.uses_contiguous_reference_output = true;
  }
  (void)tensor_io::encoded_byte_count(result.input);
  (void)tensor_io::encoded_byte_count(result.output);
  return result;
}

std::unique_ptr<BenchmarkExecutable> build_aclnn_unary_pointwise(
    const BenchmarkCase& specification) {
  return std::make_unique<AclnnUnaryBenchmarkExecutable>(
      plan_aclnn_unary_pointwise(specification));
}

}  // namespace flagdnn::benchmarking
