/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_unary_pointwise.hpp"

#include "validation/aclnn_unary_runtime.hpp"
#include "validation/functional/aclnn_binary_pointwise.hpp"
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

namespace flagdnn::testing {
namespace {

namespace acl = flagdnn::validation::ascend;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;

std::vector<std::int64_t> contiguous_strides(
    const std::vector<std::int64_t>& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    result[axis - 1] = stride;
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::overflow_error(
          "ACLNN unary pointwise contiguous stride overflows int64");
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
          "ACLNN unary pointwise binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument(
        "ACLNN unary pointwise binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument(
        "ACLNN unary pointwise binding UID " + std::to_string(uid) +
        " is not 32-byte aligned");
  }
}

acl::AclnnUnaryTensor runtime_tensor(const TestTensor& tensor) {
  return {tensor.data_type, tensor.dimensions, tensor.strides};
}

class AclnnUnaryPointwiseExecutable final : public PointwiseExecutable {
 public:
  AclnnUnaryPointwiseExecutable(PointwiseTestCase test_case,
                                AclnnUnaryPointwisePlan plan)
      : test_case_(std::move(test_case)),
        plan_(std::move(plan)),
        runtime_(test_case_.mode, test_case_.attributes) {
    validate_pointwise_case(test_case_);
  }

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
      throw AclnnPointwiseUnsupportedError(error.status(), error.what());
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
  PointwiseTestCase test_case_;
  AclnnUnaryPointwisePlan plan_;
  acl::AclnnUnaryRuntime runtime_;
};

}  // namespace

AclnnUnaryPointwisePlan plan_aclnn_unary_pointwise(
    const PointwiseTestCase& test_case) {
  validate_pointwise_case(test_case);
  if (!acl::is_aclnn_unary_mode(test_case.mode) ||
      test_case.inputs.size() != 1) {
    throw std::invalid_argument(
        "ACLNN unary pointwise requires one supported unary input");
  }
  AclnnUnaryPointwisePlan result{
      test_case.inputs.front(), test_case.output, false};
  if (result.input.data_type != result.output.data_type ||
      result.input.dimensions != result.output.dimensions) {
    throw std::invalid_argument(
        "ACLNN unary pointwise input and output metadata do not match");
  }
  const std::vector<std::int64_t> contiguous =
      contiguous_strides(result.output.dimensions);
  if (result.output.strides != contiguous) {
    result.output.strides = contiguous;
    result.uses_contiguous_reference_output = true;
  }
  (void)tensor_io::encoded_byte_count(result.input);
  (void)tensor_io::encoded_byte_count(result.output);
  return result;
}

std::unique_ptr<PointwiseExecutable> build_aclnn_unary_pointwise_reference(
    const PointwiseTestCase& test_case) {
  return std::make_unique<AclnnUnaryPointwiseExecutable>(
      test_case, plan_aclnn_unary_pointwise(test_case));
}

}  // namespace flagdnn::testing
