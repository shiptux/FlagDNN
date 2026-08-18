/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_conv_bias_relu.hpp"

#include "common/add.hpp"
#include "common/convolution.hpp"
#include "common/pointwise.hpp"
#include "validation/acl_runtime.hpp"
#include "validation/functional/aclnn_add.hpp"
#include "validation/functional/aclnn_binary_pointwise.hpp"
#include "validation/functional/aclnn_convolution.hpp"
#include "validation/functional/aclnn_unary_pointwise.hpp"
#include "validation/tensor_io.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace acl = flagdnn::validation::ascend;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;

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
          "ACLNN ConvBiasRelu binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument(
        "ACLNN ConvBiasRelu binding is missing or null");
  }
  return *result;
}

ConvolutionTestCase make_convolution_case(
    const ConvBiasReluTestCase& test_case,
    const TestTensor& convolution) {
  ConvolutionTestCase result;
  result.name = test_case.name + "_aclnn_convolution";
  result.direction = ConvolutionDirection::kFprop;
  result.x = test_case.x;
  result.w = test_case.w;
  result.y = convolution;
  result.pre_padding = test_case.padding;
  result.post_padding = test_case.padding;
  result.stride = test_case.stride;
  result.dilation = test_case.dilation;
  result.groups = 1;
  result.mode = ConvolutionMode::kCrossCorrelation;
  result.absolute_tolerance = test_case.absolute_tolerance;
  result.relative_tolerance = test_case.relative_tolerance;
  return result;
}

AddTestCase make_bias_add_case(const ConvBiasReluTestCase& test_case,
                               const TestTensor& convolution,
                               const TestTensor& biased) {
  AddTestCase result;
  result.name = test_case.name + "_aclnn_bias_add";
  result.left = convolution;
  result.right = test_case.bias;
  result.output = biased;
  result.alpha = 1.0;
  result.absolute_tolerance = test_case.absolute_tolerance;
  result.relative_tolerance = test_case.relative_tolerance;
  return result;
}

PointwiseTestCase make_relu_case(const ConvBiasReluTestCase& test_case,
                                 const TestTensor& biased) {
  PointwiseTestCase result;
  result.name = test_case.name + "_aclnn_relu";
  result.mode = FLAGDNN_POINTWISE_RELU_FWD;
  result.inputs = {biased};
  result.output = test_case.output;
  result.input_domains = {PointwiseInputDomain::kReal};
  result.absolute_tolerance = test_case.absolute_tolerance;
  result.relative_tolerance = test_case.relative_tolerance;
  return result;
}

class AclnnConvBiasReluExecutable final : public CompositeExecutable {
 public:
  AclnnConvBiasReluExecutable(ConvBiasReluTestCase test_case,
                              AclnnConvBiasReluPlan plan)
      : test_case_(std::move(test_case)), plan_(std::move(plan)) {
    validate_composite_case(test_case_);
    convolution_case_ =
        make_convolution_case(test_case_, plan_.convolution);
    bias_add_case_ =
        make_bias_add_case(test_case_, plan_.convolution, plan_.biased);
    relu_case_ = make_relu_case(test_case_, plan_.biased);
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument(
          "ACLNN ConvBiasRelu prepare stream is null");
    }
    if (state_ != nullptr) {
      acl::check_acl(
          aclrtSynchronizeStream(
              reinterpret_cast<aclrtStream>(state_->stream)),
          "aclrtSynchronizeStream(before ACLNN ConvBiasRelu reprepare)");
    }
    const flagdnnBinding_t& x = find_binding(bindings, plan_.x.uid);
    const flagdnnBinding_t& w = find_binding(bindings, plan_.w.uid);
    const flagdnnBinding_t& bias = find_binding(bindings, plan_.bias.uid);
    const flagdnnBinding_t& output =
        find_binding(bindings, plan_.output.uid);

    auto candidate = std::make_unique<State>();
    candidate->convolution_buffer = std::make_unique<acl::DeviceBuffer>(
        tensor_io::allocation_byte_count(plan_.convolution));
    candidate->biased_buffer = std::make_unique<acl::DeviceBuffer>(
        tensor_io::allocation_byte_count(plan_.biased));
    candidate->convolution = build_convolution_reference(convolution_case_);
    candidate->bias_add = build_add_reference(bias_add_case_);
    candidate->relu = build_aclnn_unary_pointwise_reference(relu_case_);

    void* const convolution_pointer =
        candidate->convolution_buffer->opaque_at(
            plan_.convolution.binding_byte_offset);
    void* const biased_pointer = candidate->biased_buffer->opaque_at(
        plan_.biased.binding_byte_offset);
    candidate->convolution_bindings = {
        {plan_.x.uid, x.device_pointer},
        {plan_.w.uid, w.device_pointer},
        {plan_.convolution.uid, convolution_pointer},
    };
    candidate->bias_add_bindings = {
        {plan_.convolution.uid, convolution_pointer},
        {plan_.bias.uid, bias.device_pointer},
        {plan_.biased.uid, biased_pointer},
    };
    candidate->relu_bindings = {
        {plan_.biased.uid, biased_pointer},
        {plan_.output.uid, output.device_pointer},
    };

    try {
      candidate->convolution->prepare(candidate->convolution_bindings,
                                      stream);
      candidate->bias_add->prepare(candidate->bias_add_bindings, stream);
      candidate->relu->prepare(candidate->relu_bindings, stream);
    } catch (const AclnnConvolutionUnsupportedError& error) {
      (void)aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
      throw AclnnConvBiasReluUnsupportedError(error.status(), error.what());
    } catch (const AclnnPointwiseUnsupportedError& error) {
      (void)aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
      throw AclnnConvBiasReluUnsupportedError(error.status(), error.what());
    } catch (const AclnnUnsupportedError& error) {
      (void)aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
      throw AclnnConvBiasReluUnsupportedError(error.status(), error.what());
    } catch (...) {
      (void)aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
      throw;
    }

    candidate->x_pointer = x.device_pointer;
    candidate->w_pointer = w.device_pointer;
    candidate->bias_pointer = bias.device_pointer;
    candidate->output_pointer = output.device_pointer;
    candidate->stream = stream;
    candidate->workspace_size = std::max(
        {candidate->convolution->workspace_size(),
         candidate->bias_add->workspace_size(),
         candidate->relu->workspace_size()});
    state_ = std::move(candidate);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return state_ == nullptr ? 0U : state_->workspace_size;
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (state_ == nullptr) {
      throw std::logic_error(
          "ACLNN ConvBiasRelu execute called before prepare");
    }
    const flagdnnBinding_t& x = find_binding(bindings, plan_.x.uid);
    const flagdnnBinding_t& w = find_binding(bindings, plan_.w.uid);
    const flagdnnBinding_t& bias = find_binding(bindings, plan_.bias.uid);
    const flagdnnBinding_t& output =
        find_binding(bindings, plan_.output.uid);
    if (x.device_pointer != state_->x_pointer ||
        w.device_pointer != state_->w_pointer ||
        bias.device_pointer != state_->bias_pointer ||
        output.device_pointer != state_->output_pointer) {
      throw std::invalid_argument(
          "ACLNN ConvBiasRelu repeatable executor binding address changed");
    }
    if (stream != state_->stream) {
      throw std::invalid_argument(
          "ACLNN ConvBiasRelu repeatable executor stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0U && workspace == nullptr)) {
      throw std::invalid_argument(
          "ACLNN ConvBiasRelu workspace is too small");
    }

    state_->convolution->execute(
        state_->convolution_bindings,
        state_->convolution->workspace_size() == 0U ? nullptr : workspace,
        state_->convolution->workspace_size(),
        stream);
    state_->bias_add->execute(
        state_->bias_add_bindings,
        state_->bias_add->workspace_size() == 0U ? nullptr : workspace,
        state_->bias_add->workspace_size(),
        stream);
    state_->relu->execute(
        state_->relu_bindings,
        state_->relu->workspace_size() == 0U ? nullptr : workspace,
        state_->relu->workspace_size(),
        stream);
  }

 private:
  struct State {
    // Executables are destroyed before buffers whose pointers they retain.
    std::unique_ptr<acl::DeviceBuffer> convolution_buffer;
    std::unique_ptr<acl::DeviceBuffer> biased_buffer;
    std::unique_ptr<ConvolutionExecutable> convolution;
    std::unique_ptr<AddExecutable> bias_add;
    std::unique_ptr<PointwiseExecutable> relu;
    std::vector<flagdnnBinding_t> convolution_bindings;
    std::vector<flagdnnBinding_t> bias_add_bindings;
    std::vector<flagdnnBinding_t> relu_bindings;
    void* x_pointer = nullptr;
    void* w_pointer = nullptr;
    void* bias_pointer = nullptr;
    void* output_pointer = nullptr;
    flagdnnStream_t stream = nullptr;
    std::size_t workspace_size = 0U;
  };

  ConvBiasReluTestCase test_case_;
  AclnnConvBiasReluPlan plan_;
  ConvolutionTestCase convolution_case_;
  AddTestCase bias_add_case_;
  PointwiseTestCase relu_case_;
  std::unique_ptr<State> state_;
};

}  // namespace

AclnnConvBiasReluPlan plan_aclnn_conv_bias_relu(
    const ConvBiasReluTestCase& test_case) {
  validate_composite_case(test_case);
  if (test_case.output.uid >
      std::numeric_limits<std::int64_t>::max() - 2) {
    throw std::overflow_error(
        "ACLNN ConvBiasRelu generated UIDs overflow int64");
  }

  TestTensor convolution = test_case.output;
  convolution.uid = test_case.output.uid + 1;
  convolution.binding_byte_offset = 0U;
  TestTensor biased = test_case.output;
  biased.uid = test_case.output.uid + 2;
  biased.binding_byte_offset = 0U;
  for (const TestTensor* external :
       {&test_case.x, &test_case.w, &test_case.bias, &test_case.output}) {
    if (convolution.uid == external->uid || biased.uid == external->uid) {
      throw std::invalid_argument(
          "ACLNN ConvBiasRelu generated UID collides with a case tensor UID");
    }
  }

  const AclnnConvolutionFpropPlan convolution_plan =
      plan_aclnn_convolution_fprop(
          make_convolution_case(test_case, convolution));
  const AclnnAddPlan bias_add_plan = plan_aclnn_add(
      make_bias_add_case(test_case, convolution_plan.output, biased));
  const AclnnUnaryPointwisePlan relu_plan =
      plan_aclnn_unary_pointwise(
          make_relu_case(test_case, bias_add_plan.output));
  (void)tensor_io::encoded_byte_count(convolution_plan.output);
  (void)tensor_io::encoded_byte_count(bias_add_plan.output);
  return {convolution_plan.input,
          convolution_plan.filter,
          bias_add_plan.right,
          convolution_plan.output,
          bias_add_plan.output,
          relu_plan.output};
}

std::unique_ptr<CompositeExecutable> build_conv_bias_relu_reference(
    const ConvBiasReluTestCase& test_case) {
  return std::make_unique<AclnnConvBiasReluExecutable>(
      test_case, plan_aclnn_conv_bias_relu(test_case));
}

}  // namespace flagdnn::testing
