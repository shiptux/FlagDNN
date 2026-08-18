/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "attention_reference.hpp"
#include "common/attention.hpp"
#include "hipdnn_reference.hpp"

#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace hv = validation::hygon;

void append_tensor(std::vector<hv::ReferenceTensor> &result,
                   const TestTensor &tensor) {
  result.push_back(hv::as_reference_tensor(tensor));
}

void append_scalar(std::vector<hv::ReferenceTensor> &result,
                   const Fp8Scalar &scalar) {
  append_tensor(result, scalar.tensor);
}

std::vector<hv::ReferenceTensor>
reference_tensors(const SdpaTestCase &test_case) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(7);
  append_tensor(result, test_case.q);
  append_tensor(result, test_case.k);
  append_tensor(result, test_case.v);
  if (test_case.bias.has_value()) {
    append_tensor(result, *test_case.bias);
  }
  append_tensor(result, test_case.output);
  if (test_case.stats.has_value()) {
    append_tensor(result, *test_case.stats);
  }
  return result;
}

std::vector<hv::ReferenceTensor>
reference_tensors(const SdpaBackwardTestCase &test_case) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(12);
  append_tensor(result, test_case.q);
  append_tensor(result, test_case.k);
  append_tensor(result, test_case.v);
  if (test_case.bias.has_value()) {
    append_tensor(result, *test_case.bias);
  }
  append_tensor(result, test_case.output);
  append_tensor(result, test_case.doutput);
  append_tensor(result, test_case.stats);
  append_tensor(result, test_case.dq);
  append_tensor(result, test_case.dk);
  append_tensor(result, test_case.dv);
  if (test_case.dbias.has_value()) {
    append_tensor(result, *test_case.dbias);
  }
  return result;
}

std::vector<hv::ReferenceTensor>
reference_tensors(const SdpaFp8TestCase &test_case) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(16);
  append_tensor(result, test_case.q);
  append_tensor(result, test_case.k);
  append_tensor(result, test_case.v);
  append_scalar(result, test_case.descale_q);
  append_scalar(result, test_case.descale_k);
  append_scalar(result, test_case.descale_v);
  append_scalar(result, test_case.descale_s);
  append_scalar(result, test_case.scale_s);
  append_scalar(result, test_case.scale_o);
  if (test_case.bias.has_value()) {
    append_tensor(result, *test_case.bias);
  }
  append_tensor(result, test_case.output);
  if (test_case.stats.has_value()) {
    append_tensor(result, *test_case.stats);
  }
  append_tensor(result, test_case.amax_s);
  append_tensor(result, test_case.amax_o);
  return result;
}

std::vector<hv::ReferenceTensor>
reference_tensors(const SdpaFp8BackwardTestCase &test_case) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(28);
  append_tensor(result, test_case.q);
  append_tensor(result, test_case.k);
  append_tensor(result, test_case.v);
  append_tensor(result, test_case.output);
  append_tensor(result, test_case.doutput);
  append_tensor(result, test_case.stats);
  append_scalar(result, test_case.descale_q);
  append_scalar(result, test_case.descale_k);
  append_scalar(result, test_case.descale_v);
  append_scalar(result, test_case.descale_o);
  append_scalar(result, test_case.descale_doutput);
  append_scalar(result, test_case.descale_s);
  append_scalar(result, test_case.descale_dp);
  append_scalar(result, test_case.scale_s);
  append_scalar(result, test_case.scale_dq);
  append_scalar(result, test_case.scale_dk);
  append_scalar(result, test_case.scale_dv);
  append_scalar(result, test_case.scale_dp);
  append_tensor(result, test_case.dq);
  append_tensor(result, test_case.dk);
  append_tensor(result, test_case.dv);
  append_tensor(result, test_case.amax_dq);
  append_tensor(result, test_case.amax_dk);
  append_tensor(result, test_case.amax_dv);
  append_tensor(result, test_case.amax_dp);
  return result;
}

template <typename Case, typename Validate, typename MakeTensors>
std::unique_ptr<AttentionExecutable>
unavailable_reference(const Case &test_case, hv::HipdnnAttentionKind kind,
                      Validate &&validate, MakeTensors &&make_tensors) {
  validate(test_case);
  const std::vector<hv::ReferenceTensor> tensors = make_tensors(test_case);
  const hv::HipdnnCapability capability = hv::hipdnn_attention_capability(
      hv::HipdnnAttentionOperation{kind}, tensors);
  hv::require_valid_hipdnn_adapter_contract(capability, "attention");
  if (capability.supported) {
    throw std::logic_error(
        "hipDNN attention capability has no executable reference plan");
  }
  throw std::invalid_argument("cannot build hipDNN attention reference: " +
                              capability.reason);
}

} // namespace

std::unique_ptr<AttentionExecutable>
build_sdpa_reference(const SdpaTestCase &test_case) {
  return unavailable_reference(
      test_case, hv::HipdnnAttentionKind::kSdpa, validate_sdpa_case,
      [](const SdpaTestCase &value) { return reference_tensors(value); });
}

std::unique_ptr<AttentionExecutable>
build_sdpa_backward_reference(const SdpaBackwardTestCase &test_case) {
  return unavailable_reference(
      test_case, hv::HipdnnAttentionKind::kSdpaBackward,
      validate_sdpa_backward_case, [](const SdpaBackwardTestCase &value) {
        return reference_tensors(value);
      });
}

std::unique_ptr<AttentionExecutable>
build_sdpa_fp8_reference(const SdpaFp8TestCase &test_case) {
  return unavailable_reference(
      test_case, hv::HipdnnAttentionKind::kSdpaFp8, validate_sdpa_fp8_case,
      [](const SdpaFp8TestCase &value) { return reference_tensors(value); });
}

std::unique_ptr<AttentionExecutable>
build_sdpa_fp8_backward_reference(const SdpaFp8BackwardTestCase &test_case) {
  return unavailable_reference(test_case,
                               hv::HipdnnAttentionKind::kSdpaFp8Backward,
                               validate_sdpa_fp8_backward_case,
                               [](const SdpaFp8BackwardTestCase &value) {
                                 return reference_tensors(value);
                               });
}

} // namespace flagdnn::testing
