/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "attention_reference.hpp"
#include "common/attention.hpp"
#include "hipdnn_reference.hpp"

#include <cstdlib>
#include <iostream>
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

constexpr int kSkipReturnCode = 77;

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

void emit_skip(const hv::HipdnnAttentionOperation &operation,
               std::string_view case_name, std::string_view reason,
               std::span<const hv::ReferenceTensor> tensors) {
  std::cout << "[SKIP][hipdnn] op="
            << hv::hipdnn_attention_kind_name(operation.kind)
            << " case=" << case_name << " reason=" << reason << ' '
            << hv::hipdnn_environment() << ' '
            << hv::describe_reference_tensors(tensors) << std::endl;
}

template <typename Case, typename Validate, typename MakeTensors>
int run_unavailable_suite(int argc, char **argv, std::span<const Case> cases,
                          std::string_view suite_name,
                          std::string_view filter_environment,
                          hv::HipdnnAttentionKind kind, Validate &&validate,
                          MakeTensors &&make_tensors) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY"
              << std::endl;
    return 2;
  }
  try {
    const char *filter = std::getenv(std::string(filter_environment).c_str());
    const hv::HipdnnAttentionOperation operation{kind};
    std::size_t matched = 0;
    std::size_t skipped = 0;
    for (const Case &test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++matched;
      validate(test_case);
      const std::vector<hv::ReferenceTensor> tensors = make_tensors(test_case);
      const hv::HipdnnCapability capability =
          hv::hipdnn_attention_capability(operation, tensors);
      hv::require_valid_hipdnn_adapter_contract(
          capability, hv::hipdnn_attention_kind_name(operation.kind));
      if (capability.supported) {
        throw std::logic_error(
            "hipDNN attention capability unexpectedly became supported");
      }
      emit_skip(operation, test_case.name, capability.reason, tensors);
      ++skipped;
    }
    if (matched == 0) {
      throw std::runtime_error(std::string(suite_name) +
                               " filter matched no test cases");
    }
    std::cout << suite_name << ": SKIP cases=" << matched
              << " executed=0 skipped=" << skipped << std::endl;
    return kSkipReturnCode;
  } catch (const std::exception &error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << std::endl;
    return 1;
  }
}

} // namespace

int run_sdpa_functional_test(int argc, char **argv,
                             std::span<const SdpaTestCase> cases) {
  return run_unavailable_suite(
      argc, argv, cases, "FLAGDNN_SDPA_FUNCTIONAL", "FLAGDNN_SDPA_CASE",
      hv::HipdnnAttentionKind::kSdpa, validate_sdpa_case,
      [](const SdpaTestCase &value) { return reference_tensors(value); });
}

int run_sdpa_backward_functional_test(
    int argc, char **argv, std::span<const SdpaBackwardTestCase> cases) {
  return run_unavailable_suite(
      argc, argv, cases, "FLAGDNN_SDPA_BACKWARD_FUNCTIONAL",
      "FLAGDNN_SDPA_BACKWARD_CASE", hv::HipdnnAttentionKind::kSdpaBackward,
      validate_sdpa_backward_case, [](const SdpaBackwardTestCase &value) {
        return reference_tensors(value);
      });
}

int run_sdpa_fp8_functional_test(int argc, char **argv,
                                 std::span<const SdpaFp8TestCase> cases) {
  return run_unavailable_suite(
      argc, argv, cases, "FLAGDNN_SDPA_FP8_FUNCTIONAL", "FLAGDNN_SDPA_FP8_CASE",
      hv::HipdnnAttentionKind::kSdpaFp8, validate_sdpa_fp8_case,
      [](const SdpaFp8TestCase &value) { return reference_tensors(value); });
}

int run_sdpa_fp8_backward_functional_test(
    int argc, char **argv, std::span<const SdpaFp8BackwardTestCase> cases) {
  return run_unavailable_suite(argc, argv, cases,
                               "FLAGDNN_SDPA_FP8_BACKWARD_FUNCTIONAL",
                               "FLAGDNN_SDPA_FP8_BACKWARD_CASE",
                               hv::HipdnnAttentionKind::kSdpaFp8Backward,
                               validate_sdpa_fp8_backward_case,
                               [](const SdpaFp8BackwardTestCase &value) {
                                 return reference_tensors(value);
                               });
}

} // namespace flagdnn::testing
