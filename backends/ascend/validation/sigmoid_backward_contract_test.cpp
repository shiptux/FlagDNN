/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/sigmoid_backward_contract.hpp"

#include <cmath>
#include <iostream>
#include <limits>

namespace sigmoid_backward =
    flagdnn::validation::ascend::sigmoid_backward_contract;

namespace {

bool close(float actual, float expected, float tolerance = 2.0e-6F) {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <= tolerance;
}

}  // namespace

int main() {
  const float log_three = std::log(3.0F);
  const float ordered = sigmoid_backward::evaluate(-4.0F, log_three);
  const float swapped = sigmoid_backward::evaluate(log_three, -4.0F);
  const float treats_logit_as_sigmoid_output =
      -4.0F * log_three * (1.0F - log_three);
  const float positive_twenty =
      sigmoid_backward::evaluate(1.0e8F, 20.0F);
  const float negative_twenty =
      sigmoid_backward::evaluate(1.0e8F, -20.0F);
  const float positive_hundred = sigmoid_backward::evaluate(
      std::numeric_limits<float>::max(), 100.0F);
  const float negative_hundred = sigmoid_backward::evaluate(
      std::numeric_limits<float>::max(), -100.0F);

  const bool stable_extremes =
      std::isfinite(sigmoid_backward::sigmoid(100.0F)) &&
      std::isfinite(sigmoid_backward::sigmoid(-100.0F)) &&
      sigmoid_backward::sigmoid(100.0F) == 1.0F &&
      sigmoid_backward::sigmoid(-100.0F) >= 0.0F;
  const bool ordered_ports_are_observable =
      std::abs(ordered - swapped) > 0.1F &&
      std::abs(ordered - treats_logit_as_sigmoid_output) > 0.1F;
  const bool stable_derivative_tail =
      close(positive_twenty, 0.206115365F) &&
      positive_twenty == negative_twenty &&
      std::isfinite(positive_hundred) && positive_hundred > 0.0F &&
      positive_hundred == negative_hundred;

  if (!close(sigmoid_backward::sigmoid(0.0F), 0.5F) ||
      !close(sigmoid_backward::evaluate(2.0F, 0.0F), 0.5F) ||
      !close(ordered, -0.75F) || !stable_extremes ||
      !ordered_ports_are_observable || !stable_derivative_tail) {
    std::cerr << "SIGMOID_BACKWARD_CONTRACT_FAILED\n";
    return 1;
  }

  std::cout << "SIGMOID_BACKWARD_CONTRACT_PASS\n";
  return 0;
}
