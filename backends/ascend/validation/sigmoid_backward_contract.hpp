/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_SIGMOID_BACKWARD_CONTRACT_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_SIGMOID_BACKWARD_CONTRACT_HPP_

#include <cmath>

namespace flagdnn::validation::ascend::sigmoid_backward_contract {

// FlagDNN's binary pointwise ports are ordered as (gradient, logit).  ACLNN's
// backward primitive instead consumes (gradient, sigmoid(logit)), so both the
// direct reference and the independent host oracle share this explicit
// semantic boundary.
[[nodiscard]] inline float sigmoid(float logit) noexcept {
  const float exponential = std::exp(-std::abs(logit));
  return logit >= 0.0F
             ? 1.0F / (1.0F + exponential)
             : exponential / (1.0F + exponential);
}

[[nodiscard]] inline float evaluate(float gradient, float logit) noexcept {
  const float exponential = std::exp(-std::abs(logit));
  const float inverse = 1.0F / (1.0F + exponential);
  return gradient * exponential * inverse * inverse;
}

}  // namespace flagdnn::validation::ascend::sigmoid_backward_contract

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_SIGMOID_BACKWARD_CONTRACT_HPP_
