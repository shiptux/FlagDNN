/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/mod_contract.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace {

namespace mod = flagdnn::validation::ascend::mod_contract;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;

bool check_type(flagdnnDataType_t data_type, std::string_view name) {
  const mod::SemanticInputs inputs = mod::semantic_inputs(data_type);
  const std::vector<float> reference =
      mod::host_reference(inputs, data_type);
  if (tensor_io::quantize(inputs.left, data_type) != inputs.left ||
      tensor_io::quantize(inputs.right, data_type) != inputs.right) {
    std::cerr << name << ": semantic inputs are not dtype-idempotent\n";
    return false;
  }

  const std::array<mod::Sentinel, 3> negative_zero = {
      mod::Sentinel::kNegativeExactPositiveDivisor,
      mod::Sentinel::kNegativeExactNegativeDivisor,
      mod::Sentinel::kNegativeZero,
  };
  for (const mod::Sentinel sentinel : negative_zero) {
    const float value = reference[mod::index(sentinel)];
    if (value != 0.0F || !std::signbit(value)) {
      std::cerr << name << ": negative-zero sentinel is invalid\n";
      return false;
    }
  }
  const std::array<mod::Sentinel, 2> positive_zero = {
      mod::Sentinel::kPositiveZero,
      mod::Sentinel::kMaximumOverHalf,
  };
  for (const mod::Sentinel sentinel : positive_zero) {
    const float value = reference[mod::index(sentinel)];
    if (value != 0.0F || std::signbit(value)) {
      std::cerr << name << ": positive-zero sentinel is invalid\n";
      return false;
    }
  }
  if (!mod::strict_zero_matches(-0.0F, -0.0F) ||
      !mod::strict_zero_matches(0.0F, 0.0F) ||
      mod::strict_zero_matches(0.0F, -0.0F) ||
      mod::strict_zero_matches(-0.0F, 0.0F) ||
      mod::strict_zero_matches(1.0e-30F, 0.0F)) {
    std::cerr << name << ": strict-zero comparator contract is invalid\n";
    return false;
  }
  if (!mod::strict_ieee_matches(-0.0F, -0.0F) ||
      !mod::strict_ieee_matches(0.0F, 0.0F) ||
      mod::strict_ieee_matches(0.0F, -0.0F) ||
      mod::strict_ieee_matches(-0.0F, 0.0F) ||
      mod::strict_ieee_matches(1.0F, std::nextafter(1.0F, 2.0F))) {
    std::cerr << name << ": strict IEEE comparator contract is invalid\n";
    return false;
  }

  const float large_nonzero =
      reference[mod::index(mod::Sentinel::kLargeNonzeroOverThree)];
  if (large_nonzero != 2.0F) {
    std::cerr << name << ": wide-exponent non-zero remainder is invalid\n";
    return false;
  }
  const float maximum_over_three =
      reference[mod::index(mod::Sentinel::kMaximumOverThree)];
  if (data_type == FLAGDNN_DATA_FLOAT16) {
    if (maximum_over_three != 2.0F) {
      std::cerr << name << ": FP16 maximum/3 remainder is invalid\n";
      return false;
    }
  } else if (maximum_over_three != 0.0F ||
             std::signbit(maximum_over_three)) {
    std::cerr << name << ": maximum/3 exact remainder is invalid\n";
    return false;
  }

  const float below =
      reference[mod::index(mod::Sentinel::kBelowExactMultiple)];
  const float above =
      reference[mod::index(mod::Sentinel::kAboveExactMultiple)];
  const float negative_above =
      reference[mod::index(mod::Sentinel::kNegativeAboveExactMultiple)];
  if (!(below > 0.0F && below < 3.0F &&
        above > 0.0F && above < 1.0F &&
        negative_above < 0.0F && negative_above > -1.0F)) {
    std::cerr << name << ": exact-multiple neighbor contract is invalid\n";
    return false;
  }

  const std::array<float, 4> signed_results = {
      reference[mod::index(
          mod::Sentinel::kNegativeDividendPositiveDivisor)],
      reference[mod::index(
          mod::Sentinel::kNegativeDividendNegativeDivisor)],
      reference[mod::index(
          mod::Sentinel::kPositiveDividendNegativeDivisor)],
      reference[mod::index(
          mod::Sentinel::kPositiveDividendPositiveDivisor)],
  };
  if (signed_results != std::array<float, 4>{-1.0F, -1.0F, 1.0F, 1.0F}) {
    std::cerr << name << ": signed-divisor contract is invalid\n";
    return false;
  }
  if (!std::all_of(reference.begin(), reference.end(), [](float value) {
        return std::isfinite(value);
      })) {
    std::cerr << name << ": semantic reference is non-finite\n";
    return false;
  }

  const mod::AclnnReferenceObservation corrected =
      mod::classify_aclnn_reference(reference, reference);
  if (corrected.matched != reference.size() || corrected.differed != 0 ||
      corrected.nonfinite != 0 || corrected.total() != reference.size()) {
    std::cerr << name
              << ": corrected ACLNN observation classification is invalid\n";
    return false;
  }

  std::vector<float> pinned_aclnn = reference;
  pinned_aclnn[mod::index(mod::Sentinel::kNegativeExactPositiveDivisor)] =
      0.0F;
  pinned_aclnn[mod::index(mod::Sentinel::kLargeNonzeroOverThree)] = 0.0F;
  pinned_aclnn[mod::index(mod::Sentinel::kMaximumOverHalf)] =
      -std::numeric_limits<float>::infinity();
  const mod::AclnnReferenceObservation limited =
      mod::classify_aclnn_reference(pinned_aclnn, reference);
  if (limited.matched != reference.size() - 3 || limited.differed != 2 ||
      limited.nonfinite != 1 || limited.total() != reference.size()) {
    std::cerr << name
              << ": pinned ACLNN limitation classification is invalid\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  const bool passed =
      check_type(FLAGDNN_DATA_FLOAT32, "fp32") &&
      check_type(FLAGDNN_DATA_FLOAT16, "fp16") &&
      check_type(FLAGDNN_DATA_BFLOAT16, "bf16");
  if (!passed) {
    return 1;
  }
  std::cout << "Ascend Mod semantic host contract passed\n";
  return 0;
}
