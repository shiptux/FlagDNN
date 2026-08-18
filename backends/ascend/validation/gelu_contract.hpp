/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_GELU_CONTRACT_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_GELU_CONTRACT_HPP_

#include <cmath>
#include <cstddef>

namespace flagdnn::validation::ascend::gelu_contract {

// Keep the production semantic check strict.  CANN 9's direct aclnnGelu
// reference can differ from the mathematical erf definition by the same scale
// as the standard tanh approximation, so only ACLNN-host and triangle checks
// receive the wider, measured reference allowance below.
inline constexpr double kFlagdnnAbsoluteTolerance = 2.0e-5;
inline constexpr double kFlagdnnRelativeTolerance = 2.0e-5;
inline constexpr double kAclnnExactAbsoluteTolerance = 5.0e-4;
inline constexpr double kAclnnExactRelativeTolerance = 5.0e-4;

inline constexpr std::size_t kScaledSentinelIndex = 21;
inline constexpr std::size_t kScaledElementCount = 24;

[[nodiscard]] inline float exact(float value) noexcept {
  constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
  return 0.5F * value *
         (1.0F + std::erf(value * kInverseSqrtTwo));
}

[[nodiscard]] inline float approximate_tanh(float value) noexcept {
  constexpr float kSqrtTwoOverPi = 0.79788456080286535588F;
  constexpr float kCubicScale = 0.044715F;
  const float cubic = value * value * value;
  return 0.5F * value *
         (1.0F + std::tanh(kSqrtTwoOverPi *
                           (value + kCubicScale * cubic)));
}

[[nodiscard]] inline float scaled_input(std::size_t index) noexcept {
  const int centered = static_cast<int>((index * 17U) % 41U) - 20;
  return static_cast<float>(centered) / 13.0F * 4.0F;
}

}  // namespace flagdnn::validation::ascend::gelu_contract

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_GELU_CONTRACT_HPP_
