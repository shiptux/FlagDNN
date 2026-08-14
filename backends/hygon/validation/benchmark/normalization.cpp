/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "ops.hpp"

#include "normalization_reference.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking::hipdnn_detail {
namespace {

namespace hv = validation::hygon;

class NormalizationExecutable final : public HipdnnExecutable {
public:
  NormalizationExecutable(hv::HipdnnNormalizationOperation operation,
                          std::vector<hv::ReferenceTensor> tensors)
      : plan_(std::move(operation), std::move(tensors)) {}

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return plan_.workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    plan_.execute(bindings, workspace, workspace_size, stream);
  }

private:
  hv::HipdnnNormalizationPlan plan_;
};

void validate_case(const BenchmarkCase &specification) {
  std::size_t expected_tensors = 0;
  std::size_t expected_outputs = 0;
  switch (specification.operation) {
  case Operation::kLayernorm:
    expected_tensors = 6;
    expected_outputs = 3;
    break;
  case Operation::kRmsnorm:
    expected_tensors = 5;
    expected_outputs = 2;
    break;
  case Operation::kBatchnorm:
    expected_tensors = 10;
    expected_outputs = 5;
    break;
  case Operation::kBatchnormInference:
    expected_tensors = 6;
    expected_outputs = 1;
    break;
  default:
    throw std::invalid_argument(
        "non-normalization case reached hipDNN normalization provider");
  }
  if (specification.tensors.size() != expected_tensors ||
      specification.output_count != expected_outputs) {
    throw std::invalid_argument(
        "normalization benchmark tensor arity is invalid");
  }
  const std::size_t inputs = input_tensor_count(specification);
  if (!specification.input_domains.empty() &&
      specification.input_domains.size() != inputs) {
    throw std::invalid_argument("normalization input domain arity is invalid");
  }
}

hv::HipdnnNormalizationOperation operation(const BenchmarkCase &specification) {
  switch (specification.operation) {
  case Operation::kLayernorm:
    return hv::make_hipdnn_normalization_unavailable(
        "hipDNN declares Other Normalization unsupported and exposes no "
        "LayerNorm primitive");
  case Operation::kRmsnorm:
    return hv::make_hipdnn_normalization_unavailable(
        "hipDNN declares Other Normalization unsupported and exposes no "
        "RMSNorm primitive");
  case Operation::kBatchnorm:
    return hv::make_hipdnn_batchnorm_training_operation(
        specification.normalization.epsilon,
        specification.normalization.momentum);
  case Operation::kBatchnormInference:
    return hv::make_hipdnn_normalization_unavailable(
        "FlagDNN consumes inverse variance directly, while hipDNN inference "
        "consumes variance plus epsilon; the primitive equations are not "
        "input-equivalent");
  default:
    return hv::make_hipdnn_normalization_unavailable(
        "operation is outside the Hygon normalization adapter");
  }
}

TensorSpec dense_tensor(const TensorSpec &tensor) {
  const hv::ReferenceTensor dense =
      hv::batchnorm_dense_reference_tensor(hv::as_reference_tensor(tensor));
  return {dense.uid, dense.data_type, dense.dimensions, dense.strides,
          dense.binding_byte_offset};
}

std::vector<hv::ReferenceTensor>
as_reference_tensors(std::span<const TensorSpec> tensors) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(tensors.size());
  for (const TensorSpec &tensor : tensors) {
    result.push_back(hv::as_reference_tensor(tensor));
  }
  return result;
}

} // namespace

std::vector<TensorSpec>
normalization_reference_specs(const BenchmarkCase &specification) {
  validate_case(specification);
  std::vector<TensorSpec> result = specification.tensors;
  if ((specification.operation == Operation::kBatchnorm ||
       specification.operation == Operation::kBatchnormInference) &&
      !result.empty()) {
    const std::size_t inputs = input_tensor_count(specification);
    result[0] = dense_tensor(result[0]);
    result[inputs] = dense_tensor(result[inputs]);
  }
  return result;
}

std::vector<hv::ReferenceTensor>
normalization_diagnostic_tensors(const BenchmarkCase &specification) {
  return as_reference_tensors(normalization_reference_specs(specification));
}

ProviderCapability
normalization_capability(const BenchmarkCase &specification) {
  validate_case(specification);
  const hv::HipdnnCapability result = hv::hipdnn_normalization_capability(
      operation(specification),
      normalization_diagnostic_tensors(specification));
  hv::require_valid_hipdnn_adapter_contract(
      result, normalization_operation_name(specification));
  return {result.supported, result.reason};
}

std::unique_ptr<HipdnnExecutable>
build_normalization(const BenchmarkCase &specification) {
  const ProviderCapability support = normalization_capability(specification);
  if (!support.supported) {
    throw BenchmarkUnsupportedError(support.reason);
  }
  return std::make_unique<NormalizationExecutable>(
      operation(specification),
      normalization_diagnostic_tensors(specification));
}

std::string normalization_operation_name(const BenchmarkCase &specification) {
  switch (specification.operation) {
  case Operation::kLayernorm:
    return "layernorm";
  case Operation::kRmsnorm:
    return "rmsnorm";
  case Operation::kBatchnorm:
    return "batchnorm";
  case Operation::kBatchnormInference:
    return "batchnorm_inference";
  default:
    return "normalization";
  }
}

} // namespace flagdnn::benchmarking::hipdnn_detail
