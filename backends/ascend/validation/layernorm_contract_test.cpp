/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/layernorm_validation.hpp"

#include "validation/functional/aclnn_layernorm.hpp"
#include "validation/benchmark/aclnn_layernorm_provider.hpp"

#include "common/cases.hpp"

#include <acl/acl.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

template <typename Function>
void require_invalid(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

void check_catalog_and_plan() {
  const auto common = flagdnn::testing::make_layernorm_cases();
  const auto cases = flagdnn::validation::ascend::make_ascend_layernorm_cases(
      common);
  if (common.size() != 9U || cases.size() != 9U) {
    throw std::runtime_error(
        "Ascend LayerNorm functional catalog is incomplete");
  }
  const auto benchmark_common =
      flagdnn::benchmarking::layernorm_benchmark_cases();
  const auto benchmark =
      flagdnn::validation::ascend::make_ascend_layernorm_benchmark_cases(
          benchmark_common);
  if (benchmark_common.size() != 15U || benchmark.size() != 15U) {
    throw std::runtime_error(
        "Ascend LayerNorm benchmark catalog is incomplete");
  }

  const auto plan = flagdnn::validation::ascend::plan_layernorm(cases.front());
  if (plan.rows != 10 || plan.normalized_elements != 17 ||
      plan.input.uid != cases.front().x.uid ||
      plan.output.uid != cases.front().y.uid ||
      plan.mean.uid != cases.front().mean.uid ||
      plan.inverse_variance.uid != cases.front().inv_variance.uid ||
      plan.epsilon != cases.front().epsilon) {
    throw std::runtime_error("Ascend LayerNorm plan decomposition is invalid");
  }
  const auto benchmark_plan =
      flagdnn::validation::ascend::plan_layernorm(benchmark.front());
  if (benchmark_plan.rows != 128 ||
      benchmark_plan.normalized_elements != 768) {
    throw std::runtime_error("Ascend LayerNorm benchmark plan is invalid");
  }

  const auto reference =
      flagdnn::testing::plan_aclnn_layernorm(cases.front());
  if (reference.operation.rows != 10 ||
      reference.normalized_shape != std::vector<std::int64_t>{17}) {
    throw std::runtime_error("ACLNN LayerNorm reference plan is invalid");
  }
  const auto executable =
      flagdnn::testing::build_layernorm_reference(cases.front());
  if (executable == nullptr || executable->workspace_size() != 0U) {
    throw std::runtime_error(
        "ACLNN LayerNorm reference has invalid pre-prepare state");
  }
  const auto benchmark_executable =
      flagdnn::benchmarking::build_aclnn_layernorm(benchmark.front());
  if (benchmark_executable == nullptr ||
      benchmark_executable->workspace_size() != 0U) {
    throw std::runtime_error(
        "ACLNN LayerNorm benchmark provider has invalid pre-prepare state");
  }
}

void check_host_oracle() {
  const auto test_case = flagdnn::testing::make_layernorm_cases().front();
  const auto plan = flagdnn::validation::ascend::plan_layernorm(test_case);
  const std::size_t elements = static_cast<std::size_t>(
      plan.rows * plan.normalized_elements);
  const std::size_t normalized =
      static_cast<std::size_t>(plan.normalized_elements);
  std::vector<float> input(elements);
  std::vector<float> scale(normalized);
  std::vector<float> bias(normalized);
  for (std::size_t index = 0; index < elements; ++index) {
    input[index] = static_cast<float>(static_cast<int>(index % 13U) - 6) /
                   8.0F;
  }
  for (std::size_t index = 0; index < normalized; ++index) {
    scale[index] = index % 2U == 0U ? 1.25F : -0.75F;
    bias[index] = static_cast<float>(index % 5U) * 0.125F - 0.25F;
  }
  std::vector<float> output(elements, 0.0F);
  std::vector<float> mean(static_cast<std::size_t>(plan.rows), 0.0F);
  std::vector<float> inverse_variance(static_cast<std::size_t>(plan.rows),
                                      0.0F);
  flagdnn::validation::ascend::layernorm_host_oracle(
      plan, input, scale, bias, output, mean, inverse_variance);

  double expected_mean = 0.0;
  for (std::size_t column = 0; column < normalized; ++column) {
    expected_mean += input[column];
  }
  expected_mean /= static_cast<double>(normalized);
  double variance = 0.0;
  for (std::size_t column = 0; column < normalized; ++column) {
    const double centered = input[column] - expected_mean;
    variance += centered * centered;
  }
  variance /= static_cast<double>(normalized);
  const double expected_inverse = 1.0 / std::sqrt(variance + plan.epsilon);
  const double expected_output =
      (input[0] - expected_mean) * expected_inverse * scale[0] + bias[0];
  if (std::abs(mean[0] - expected_mean) > 1.0e-6 ||
      std::abs(inverse_variance[0] - expected_inverse) > 1.0e-6 ||
      std::abs(output[0] - expected_output) > 1.0e-6) {
    throw std::runtime_error("Ascend LayerNorm host oracle formula is invalid");
  }
}

void check_rejections() {
  auto test_case = flagdnn::testing::make_layernorm_cases().front();
  test_case.mean.data_type = FLAGDNN_DATA_FLOAT16;
  require_invalid(
      [&] { (void)flagdnn::validation::ascend::plan_layernorm(test_case); },
      "LayerNorm non-FP32 mean was accepted");

  test_case = flagdnn::testing::make_layernorm_cases().front();
  test_case.scale.dimensions = {5, 17};
  test_case.scale.strides = {17, 1};
  require_invalid(
      [&] { (void)flagdnn::validation::ascend::plan_layernorm(test_case); },
      "LayerNorm nonsuffix scale shape was accepted");

  test_case = flagdnn::testing::make_layernorm_cases().front();
  test_case.mean.uid = test_case.y.uid;
  require_invalid(
      [&] { (void)flagdnn::validation::ascend::plan_layernorm(test_case); },
      "LayerNorm aliased binding UID was accepted");

  test_case = flagdnn::testing::make_layernorm_cases().front();
  test_case.epsilon = -1.0;
  require_invalid(
      [&] { (void)flagdnn::validation::ascend::plan_layernorm(test_case); },
      "LayerNorm negative epsilon was accepted");
}

}  // namespace

int main() {
  if (!flagdnn::testing::aclnn_layernorm_status_is_unsupported(
          ACL_ERROR_API_NOT_SUPPORT, "ACLNN LayerNorm is not supported") ||
      flagdnn::testing::aclnn_layernorm_status_is_unsupported(
          123456, "invalid LayerNorm descriptor")) {
    throw std::runtime_error(
        "ACLNN LayerNorm unsupported status classification is invalid");
  }
  check_catalog_and_plan();
  check_host_oracle();
  check_rejections();
  return 0;
}
