/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "ops.hpp"

#include "convolution_reference.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking::hipdnn_detail {
namespace {

namespace hv = validation::hygon;

hv::HipdnnConvolutionAlgorithmPolicy
algorithm_policy(HipdnnReferencePolicy policy) {
  switch (policy) {
  case HipdnnReferencePolicy::kCorrectnessOracle:
    return hv::HipdnnConvolutionAlgorithmPolicy::kCorrectnessOracle;
  case HipdnnReferencePolicy::kPerformance:
    return hv::HipdnnConvolutionAlgorithmPolicy::kPerformance;
  }
  throw std::invalid_argument("invalid hipDNN benchmark reference policy");
}

class ConvolutionExecutable final : public HipdnnExecutable {
public:
  ConvolutionExecutable(hv::HipdnnConvolutionOperation operation,
                        std::vector<hv::ReferenceTensor> tensors,
                        HipdnnReferencePolicy policy)
      : plan_(std::move(operation), std::move(tensors),
              algorithm_policy(policy)) {}

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return plan_.workspace_size();
  }

  [[nodiscard]] ProviderCapability
  probe(std::span<const flagdnnBinding_t> bindings, void *workspace,
        std::size_t workspace_size, flagdnnStream_t stream) override {
    const hv::HipdnnCapability setup = plan_.capability();
    hv::require_valid_hipdnn_adapter_contract(setup,
                                              "benchmark convolution setup");
    if (!setup.supported) {
      return ProviderCapability::unsupported(setup.reason);
    }
    const hv::HipdnnCapability execution =
        plan_.probe_execute(bindings, workspace, workspace_size, stream);
    hv::require_valid_hipdnn_adapter_contract(
        execution, "benchmark convolution execute probe");
    return {execution.supported, execution.reason};
  }

  [[nodiscard]] std::string runtime_description() const override {
    const std::string_view algorithm = plan_.selected_algorithm_name();
    if (algorithm.empty()) {
      return {};
    }
    return "policy=" +
           std::string(
               hv::hipdnn_convolution_algorithm_policy_name(plan_.policy())) +
           " algorithm=" + std::string(algorithm);
  }

  void reject_selected_candidate(std::string reason) override {
    plan_.reject_selected_algorithm(std::move(reason));
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    plan_.execute(bindings, workspace, workspace_size, stream);
  }

private:
  hv::HipdnnConvolutionPlan plan_;
};

hv::HipdnnConvolutionMode reference_mode(ConvolutionMode mode) noexcept {
  return mode == ConvolutionMode::kConvolution
             ? hv::HipdnnConvolutionMode::kConvolution
             : hv::HipdnnConvolutionMode::kCrossCorrelation;
}

bool is_conv_bias_relu(const BenchmarkCase &specification) noexcept {
  if (specification.operation != Operation::kGraph ||
      specification.output_count != 1 || specification.tensors.size() != 4 ||
      specification.graph.intermediates.size() != 2 ||
      specification.graph.nodes.size() != 3) {
    return false;
  }
  const GraphNodeSpec &convolution = specification.graph.nodes[0];
  const GraphNodeSpec &bias = specification.graph.nodes[1];
  const GraphNodeSpec &relu = specification.graph.nodes[2];
  const TensorSpec &output = specification.tensors[3];
  const auto matches_output_metadata = [&](const TensorSpec &tensor) {
    return tensor.data_type == output.data_type &&
           tensor.dimensions == output.dimensions &&
           tensor.strides == output.strides;
  };
  return convolution.operation == Operation::kConvolutionFprop &&
         convolution.convolution.mode == ConvolutionMode::kCrossCorrelation &&
         convolution.input_uids.size() == 2 &&
         convolution.input_uids[0] == specification.tensors[0].uid &&
         convolution.input_uids[1] == specification.tensors[1].uid &&
         convolution.output_uid == specification.graph.intermediates[0].uid &&
         bias.operation == Operation::kPointwise &&
         bias.pointwise_mode == FLAGDNN_POINTWISE_ADD && bias.alpha == 1.0 &&
         bias.pointwise_attributes.flags == 0U && bias.input_uids.size() == 2 &&
         bias.input_uids[0] == convolution.output_uid &&
         bias.input_uids[1] == specification.tensors[2].uid &&
         bias.output_uid == specification.graph.intermediates[1].uid &&
         relu.operation == Operation::kPointwise &&
         relu.pointwise_mode == FLAGDNN_POINTWISE_RELU_FWD &&
         relu.alpha == 1.0 && relu.pointwise_attributes.flags == 0U &&
         relu.input_uids.size() == 1 && relu.input_uids[0] == bias.output_uid &&
         relu.output_uid == output.uid &&
         matches_output_metadata(specification.graph.intermediates[0]) &&
         matches_output_metadata(specification.graph.intermediates[1]);
}

hv::HipdnnConvolutionOperation operation(const BenchmarkCase &specification) {
  if (specification.operation == Operation::kConvolutionFprop ||
      specification.operation == Operation::kConvolutionDgrad ||
      specification.operation == Operation::kConvolutionWgrad) {
    hv::HipdnnConvolutionKind kind = hv::HipdnnConvolutionKind::kFprop;
    if (specification.operation == Operation::kConvolutionDgrad) {
      kind = hv::HipdnnConvolutionKind::kDgrad;
    } else if (specification.operation == Operation::kConvolutionWgrad) {
      kind = hv::HipdnnConvolutionKind::kWgrad;
    }
    return hv::make_hipdnn_convolution_operation(
        kind, specification.convolution.pre_padding,
        specification.convolution.post_padding,
        specification.convolution.stride, specification.convolution.dilation,
        specification.convolution.groups,
        reference_mode(specification.convolution.mode));
  }
  if (is_conv_bias_relu(specification)) {
    const ConvolutionAttributes &attributes =
        specification.graph.nodes[0].convolution;
    return hv::make_hipdnn_convolution_operation(
        hv::HipdnnConvolutionKind::kConvBiasRelu, attributes.pre_padding,
        attributes.post_padding, attributes.stride, attributes.dilation,
        attributes.groups, reference_mode(attributes.mode));
  }
  return hv::make_hipdnn_convolution_unavailable(
      "benchmark graph is not an exact public ConvBiasRelu sequence");
}

std::vector<hv::ReferenceTensor>
semantic_reference_tensors(const BenchmarkCase &specification,
                           std::span<const TensorSpec> tensors) {
  if (tensors.size() != specification.tensors.size()) {
    throw std::invalid_argument(
        "hipDNN convolution policy changed the tensor arity");
  }
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    if (tensors[index].uid != specification.tensors[index].uid) {
      throw std::invalid_argument(
          "hipDNN convolution policy changed a semantic tensor UID");
    }
  }
  if (specification.output_count != 1) {
    return {};
  }
  if (specification.operation == Operation::kConvolutionFprop &&
      tensors.size() == 3) {
    return {hv::as_reference_tensor(tensors[0]),
            hv::as_reference_tensor(tensors[1]),
            hv::as_reference_tensor(tensors[2])};
  }
  if (specification.operation == Operation::kConvolutionDgrad &&
      tensors.size() == 3) {
    return {hv::as_reference_tensor(tensors[2]),
            hv::as_reference_tensor(tensors[1]),
            hv::as_reference_tensor(tensors[0])};
  }
  if (specification.operation == Operation::kConvolutionWgrad &&
      tensors.size() == 3) {
    return {hv::as_reference_tensor(tensors[1]),
            hv::as_reference_tensor(tensors[2]),
            hv::as_reference_tensor(tensors[0])};
  }
  if (is_conv_bias_relu(specification)) {
    return {hv::as_reference_tensor(tensors[0]),
            hv::as_reference_tensor(tensors[1]),
            hv::as_reference_tensor(tensors[3]),
            hv::as_reference_tensor(tensors[2])};
  }
  return {};
}

TensorSpec fp32_oracle_tensor(const TensorSpec &source) {
  constexpr std::size_t kSourceElementBytes = sizeof(std::uint16_t);
  constexpr std::size_t kOracleElementBytes = sizeof(float);
  static_assert(kSourceElementBytes == 2);
  static_assert(kOracleElementBytes == 4);

  if (source.data_type != FLAGDNN_DATA_FLOAT16) {
    throw std::invalid_argument(
        "FP32 convolution oracle requires a source FP16 tensor");
  }
  if (source.binding_byte_offset % kSourceElementBytes != 0) {
    throw std::invalid_argument(
        "FP16 convolution binding byte offset is not element aligned");
  }
  const std::size_t offset_elements =
      source.binding_byte_offset / kSourceElementBytes;
  if (offset_elements >
      std::numeric_limits<std::size_t>::max() / kOracleElementBytes) {
    throw std::overflow_error(
        "FP32 convolution oracle binding byte offset overflows");
  }

  TensorSpec result = source;
  result.data_type = FLAGDNN_DATA_FLOAT32;
  result.binding_byte_offset = offset_elements * kOracleElementBytes;
  return result;
}

} // namespace

bool is_convolution_case(const BenchmarkCase &specification) noexcept {
  return specification.operation == Operation::kConvolutionFprop ||
         specification.operation == Operation::kConvolutionDgrad ||
         specification.operation == Operation::kConvolutionWgrad ||
         is_conv_bias_relu(specification);
}

ProviderCapability convolution_capability(const BenchmarkCase &specification) {
  const hv::HipdnnCapability result = hv::hipdnn_convolution_capability(
      operation(specification),
      semantic_reference_tensors(specification, specification.tensors));
  hv::require_valid_hipdnn_adapter_contract(
      result, convolution_operation_name(specification));
  return {result.supported, result.reason};
}

std::unique_ptr<HipdnnExecutable>
build_convolution(const BenchmarkCase &specification,
                  HipdnnReferencePolicy policy) {
  const ProviderCapability support = convolution_capability(specification);
  if (!support.supported) {
    throw BenchmarkUnsupportedError(support.reason);
  }
  const std::vector<TensorSpec> reference_specs =
      convolution_reference_specs(specification, policy);
  return std::make_unique<ConvolutionExecutable>(
      operation(specification),
      semantic_reference_tensors(specification, reference_specs), policy);
}

bool convolution_uses_fp32_correctness_oracle(
    const BenchmarkCase &specification) noexcept {
  return is_convolution_case(specification) &&
         !specification.tensors.empty() &&
         std::all_of(specification.tensors.begin(), specification.tensors.end(),
                     [](const TensorSpec &tensor) {
                       return tensor.data_type == FLAGDNN_DATA_FLOAT16;
                     });
}

std::vector<TensorSpec>
convolution_reference_specs(const BenchmarkCase &specification,
                            HipdnnReferencePolicy policy) {
  if (policy != HipdnnReferencePolicy::kCorrectnessOracle &&
      policy != HipdnnReferencePolicy::kPerformance) {
    throw std::invalid_argument("invalid hipDNN convolution reference policy");
  }
  std::vector<TensorSpec> result = specification.tensors;
  if (policy == HipdnnReferencePolicy::kPerformance ||
      !convolution_uses_fp32_correctness_oracle(specification)) {
    return result;
  }
  std::transform(result.begin(), result.end(), result.begin(),
                 fp32_oracle_tensor);
  return result;
}

std::vector<hv::ReferenceTensor> convolution_semantic_reference_tensors(
    const BenchmarkCase &specification, HipdnnReferencePolicy policy) {
  const std::vector<TensorSpec> reference_specs =
      convolution_reference_specs(specification, policy);
  return semantic_reference_tensors(specification, reference_specs);
}

std::vector<hv::ReferenceTensor>
convolution_diagnostic_tensors(const BenchmarkCase &specification) {
  return semantic_reference_tensors(specification, specification.tensors);
}

std::string convolution_operation_name(const BenchmarkCase &specification) {
  switch (specification.operation) {
  case Operation::kConvolutionFprop:
    return "conv_fprop";
  case Operation::kConvolutionDgrad:
    return "conv_dgrad";
  case Operation::kConvolutionWgrad:
    return "conv_wgrad";
  case Operation::kGraph:
    return is_conv_bias_relu(specification) ? "conv_bias_relu" : "graph";
  default:
    return "convolution";
  }
}

bool convolution_uses_sequence(const BenchmarkCase &specification) noexcept {
  return is_conv_bias_relu(specification);
}

} // namespace flagdnn::benchmarking::hipdnn_detail
