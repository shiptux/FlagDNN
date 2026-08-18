/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "hipdnn_provider.hpp"

#include "ops.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

enum class Family {
  kPointwise,
  kTensor,
  kConvolution,
  kNormalization,
};

Family family(const BenchmarkCase &specification) {
  switch (specification.operation) {
  case Operation::kRelu:
  case Operation::kPointwise:
  case Operation::kAdd:
    return Family::kPointwise;
  case Operation::kReduction:
  case Operation::kMatmul:
  case Operation::kReshape:
  case Operation::kTranspose:
  case Operation::kSlice:
    return Family::kTensor;
  case Operation::kConvolutionFprop:
  case Operation::kConvolutionDgrad:
  case Operation::kConvolutionWgrad:
    return Family::kConvolution;
  case Operation::kLayernorm:
  case Operation::kRmsnorm:
  case Operation::kBatchnorm:
  case Operation::kBatchnormInference:
    return Family::kNormalization;
  case Operation::kGraph:
    return hipdnn_detail::is_convolution_case(specification)
               ? Family::kConvolution
               : Family::kPointwise;
  }
  throw std::invalid_argument("unknown benchmark operation family");
}

} // namespace

ProviderCapability
HipdnnProvider::capability(const BenchmarkCase &specification) const {
  switch (family(specification)) {
  case Family::kPointwise:
    return hipdnn_detail::pointwise_capability(specification);
  case Family::kTensor:
    return hipdnn_detail::tensor_capability(specification);
  case Family::kConvolution:
    return hipdnn_detail::convolution_capability(specification);
  case Family::kNormalization:
    return hipdnn_detail::normalization_capability(specification);
  }
  throw std::invalid_argument("unknown hipDNN benchmark family");
}

ProviderCapability
HipdnnProvider::capture_capability(const BenchmarkCase &specification) const {
  if (family(specification) == Family::kPointwise) {
    return hipdnn_detail::pointwise_capture_capability(specification);
  }
  return {};
}

std::unique_ptr<HipdnnExecutable>
HipdnnProvider::build_reference(const BenchmarkCase &specification,
                                HipdnnReferencePolicy policy) const {
  switch (family(specification)) {
  case Family::kPointwise:
    return hipdnn_detail::build_pointwise(specification);
  case Family::kTensor:
    return hipdnn_detail::build_tensor(specification);
  case Family::kConvolution:
    return hipdnn_detail::build_convolution(specification, policy);
  case Family::kNormalization:
    return hipdnn_detail::build_normalization(specification);
  }
  throw std::invalid_argument("unknown hipDNN benchmark family");
}

std::unique_ptr<BenchmarkExecutable>
HipdnnProvider::build(const BenchmarkCase &specification) {
  return build_reference(specification,
                         HipdnnReferencePolicy::kCorrectnessOracle);
}

bool HipdnnProvider::requires_accuracy_gated_performance(
    const BenchmarkCase &specification) const {
  return family(specification) == Family::kConvolution;
}

bool HipdnnProvider::uses_fp32_correctness_oracle(
    const BenchmarkCase &specification) const {
  return family(specification) == Family::kConvolution &&
         hipdnn_detail::convolution_uses_fp32_correctness_oracle(specification);
}

std::vector<TensorSpec>
HipdnnProvider::reference_specs(const BenchmarkCase &specification,
                                HipdnnReferencePolicy policy) const {
  switch (family(specification)) {
  case Family::kPointwise:
    return hipdnn_detail::pointwise_reference_specs(specification);
  case Family::kTensor:
    return hipdnn_detail::tensor_reference_specs(specification);
  case Family::kConvolution:
    return hipdnn_detail::convolution_reference_specs(specification, policy);
  case Family::kNormalization:
    return hipdnn_detail::normalization_reference_specs(specification);
  }
  throw std::invalid_argument("unknown hipDNN benchmark family");
}

std::vector<validation::hygon::ReferenceTensor>
HipdnnProvider::diagnostic_tensors(const BenchmarkCase &specification) const {
  switch (family(specification)) {
  case Family::kPointwise:
    return hipdnn_detail::pointwise_diagnostic_tensors(specification);
  case Family::kTensor:
    return hipdnn_detail::tensor_diagnostic_tensors(specification);
  case Family::kConvolution:
    return hipdnn_detail::convolution_diagnostic_tensors(specification);
  case Family::kNormalization:
    return hipdnn_detail::normalization_diagnostic_tensors(specification);
  }
  throw std::invalid_argument("unknown hipDNN benchmark family");
}

std::string
HipdnnProvider::operation_name(const BenchmarkCase &specification) const {
  switch (family(specification)) {
  case Family::kPointwise:
    return hipdnn_detail::pointwise_operation_name(specification);
  case Family::kTensor:
    return hipdnn_detail::tensor_operation_name(specification);
  case Family::kConvolution:
    return hipdnn_detail::convolution_operation_name(specification);
  case Family::kNormalization:
    return hipdnn_detail::normalization_operation_name(specification);
  }
  throw std::invalid_argument("unknown hipDNN benchmark family");
}

bool HipdnnProvider::uses_sequence(
    const BenchmarkCase &specification) const {
  switch (family(specification)) {
  case Family::kPointwise:
    return hipdnn_detail::pointwise_uses_sequence(specification);
  case Family::kConvolution:
    return hipdnn_detail::convolution_uses_sequence(specification);
  case Family::kTensor:
  case Family::kNormalization:
    return false;
  }
  return false;
}

} // namespace flagdnn::benchmarking
