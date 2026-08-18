/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/gelu_contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

namespace gelu = flagdnn::validation::ascend::gelu_contract;

int main() {
  double maximum_absolute = 0.0;
  std::size_t maximum_index = 0;
  std::size_t strict_failures = 0;
  for (std::size_t index = 0; index < gelu::kScaledElementCount; ++index) {
    const float value = gelu::scaled_input(index);
    const double exact = gelu::exact(value);
    const double approximate = gelu::approximate_tanh(value);
    const double absolute = std::abs(exact - approximate);
    const double relative =
        absolute / std::max({std::abs(exact), std::abs(approximate), 1.0e-30});
    if (absolute > maximum_absolute) {
      maximum_absolute = absolute;
      maximum_index = index;
    }
    if (absolute > gelu::kFlagdnnAbsoluteTolerance &&
        relative > gelu::kFlagdnnRelativeTolerance) {
      ++strict_failures;
    }
  }

  const float sentinel = gelu::scaled_input(gelu::kScaledSentinelIndex);
  const double sentinel_exact = gelu::exact(sentinel);
  const double sentinel_approximate = gelu::approximate_tanh(sentinel);
  const double sentinel_absolute =
      std::abs(sentinel_exact - sentinel_approximate);
  const double sentinel_relative = sentinel_absolute /
      std::max({std::abs(sentinel_exact),
                std::abs(sentinel_approximate),
                1.0e-30});

  const bool sentinel_rejects_mode_swap =
      sentinel_absolute > gelu::kFlagdnnAbsoluteTolerance &&
      sentinel_relative > gelu::kFlagdnnRelativeTolerance;
  const bool aclnn_allowance_covers_measured_approximation =
      maximum_absolute <= gelu::kAclnnExactAbsoluteTolerance;
  if (maximum_index != gelu::kScaledSentinelIndex ||
      strict_failures == 0 || !sentinel_rejects_mode_swap ||
      !aclnn_allowance_covers_measured_approximation ||
      gelu::kAclnnExactAbsoluteTolerance <=
          gelu::kFlagdnnAbsoluteTolerance) {
    std::cerr << "Ascend GELU validation tolerance contract failed"
              << " max_abs=" << maximum_absolute
              << " max_index=" << maximum_index
              << " strict_failures=" << strict_failures << '\n';
    return 1;
  }

  std::cout << "Ascend GELU validation tolerance contract passed"
            << " sentinel=" << sentinel
            << " max_abs=" << maximum_absolute
            << " strict_failures=" << strict_failures << '\n';
  return 0;
}
