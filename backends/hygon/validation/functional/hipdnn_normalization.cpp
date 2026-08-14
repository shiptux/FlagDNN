/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/normalization.hpp"
#include "normalization_reference.hpp"

#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace hv = validation::hygon;

class ReferenceExecutable final : public NormalizationExecutable {
public:
  ReferenceExecutable(hv::HipdnnNormalizationOperation operation,
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

std::vector<hv::ReferenceTensor>
as_reference_tensors(std::span<const TestTensor> tensors) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(tensors.size());
  for (const TestTensor &tensor : tensors) {
    result.push_back(hv::as_reference_tensor(tensor));
  }
  return result;
}

[[noreturn]] void throw_unavailable(std::string_view operation_name) {
  throw std::runtime_error(std::string("hipDNN has no exact ") +
                           std::string(operation_name) + " primitive");
}

} // namespace

TestTensor batchnorm_reference_data_tensor(const TestTensor &tensor) {
  const hv::ReferenceTensor dense =
      hv::batchnorm_dense_reference_tensor(hv::as_reference_tensor(tensor));
  return {dense.uid, dense.data_type, dense.dimensions, dense.strides,
          dense.binding_byte_offset};
}

std::unique_ptr<NormalizationExecutable>
build_layernorm_reference(const LayernormTestCase &test_case) {
  validate_normalization_case(test_case);
  throw_unavailable("LayerNorm");
}

std::unique_ptr<NormalizationExecutable>
build_rmsnorm_reference(const RmsnormTestCase &test_case) {
  validate_normalization_case(test_case);
  throw_unavailable("RMSNorm");
}

std::unique_ptr<NormalizationExecutable>
build_batchnorm_reference(const BatchnormTestCase &test_case) {
  validate_normalization_case(test_case);
  std::vector<TestTensor> specifications = {
      batchnorm_reference_data_tensor(test_case.x),
      test_case.scale,
      test_case.bias,
      test_case.previous_running_mean,
      test_case.previous_running_variance,
      batchnorm_reference_data_tensor(test_case.y),
      test_case.mean,
      test_case.inv_variance,
      test_case.next_running_mean,
      test_case.next_running_variance};
  return std::make_unique<ReferenceExecutable>(
      hv::make_hipdnn_batchnorm_training_operation(test_case.epsilon,
                                                   test_case.momentum),
      as_reference_tensors(specifications));
}

std::unique_ptr<NormalizationExecutable> build_batchnorm_inference_reference(
    const BatchnormInferenceTestCase &test_case) {
  validate_normalization_case(test_case);
  throw_unavailable("BatchNorm Inference with inverse-variance input");
}

} // namespace flagdnn::testing
