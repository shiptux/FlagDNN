/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/rmsnorm_validation.hpp"

#include "validation/functional/aclnn_rmsnorm.hpp"
#include "validation/benchmark/aclnn_rmsnorm_provider.hpp"

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
  const auto common = flagdnn::testing::make_rmsnorm_cases();
  const auto cases = flagdnn::validation::ascend::make_ascend_rmsnorm_cases(
      common);
  if (common.size() != 9U || cases.size() != 9U) {
    throw std::runtime_error("Ascend RMSNorm functional catalog is incomplete");
  }
  const auto benchmark_common =
      flagdnn::benchmarking::rmsnorm_benchmark_cases();
  const auto benchmark =
      flagdnn::validation::ascend::make_ascend_rmsnorm_benchmark_cases(
          benchmark_common);
  if (benchmark_common.size() != 15U || benchmark.size() != 15U) {
    throw std::runtime_error("Ascend RMSNorm benchmark catalog is incomplete");
  }

  const auto plan = flagdnn::validation::ascend::plan_rmsnorm(cases.front());
  if (plan.rows != 10 || plan.normalized_elements != 17 ||
      plan.input.uid != cases.front().x.uid ||
      plan.output.uid != cases.front().y.uid ||
      plan.inverse_variance.uid != cases.front().inv_variance.uid ||
      plan.epsilon != cases.front().epsilon) {
    throw std::runtime_error("Ascend RMSNorm plan decomposition is invalid");
  }
  const auto benchmark_plan =
      flagdnn::validation::ascend::plan_rmsnorm(benchmark.front());
  if (benchmark_plan.rows != 128 ||
      benchmark_plan.normalized_elements != 768) {
    throw std::runtime_error("Ascend RMSNorm benchmark plan is invalid");
  }

  const auto reference = flagdnn::testing::plan_aclnn_rmsnorm(cases.front());
  if (reference.operation.rows != 10 ||
      reference.gamma.dimensions != std::vector<std::int64_t>{17} ||
      reference.gamma.strides != std::vector<std::int64_t>{1} ||
      reference.gamma.uid != cases.front().scale.uid ||
      reference.normalized_output.uid != 0 ||
      reference.normalized_output.data_type != cases.front().y.data_type ||
      reference.normalized_output.dimensions != cases.front().y.dimensions ||
      reference.normalized_output.strides != cases.front().y.strides ||
      reference.normalized_output.binding_byte_offset != 0 ||
      reference.normalized_output_bytes == 0) {
    throw std::runtime_error(
        "ACLNN RMSNorm composite reference plan is invalid");
  }
  const auto executable =
      flagdnn::testing::build_rmsnorm_reference(cases.front());
  if (executable == nullptr || executable->workspace_size() != 0U) {
    throw std::runtime_error(
        "ACLNN RMSNorm reference has invalid pre-prepare state");
  }
  const auto benchmark_executable =
      flagdnn::benchmarking::build_aclnn_rmsnorm(benchmark.front());
  if (benchmark_executable == nullptr ||
      benchmark_executable->workspace_size() != 0U) {
    throw std::runtime_error(
        "ACLNN RMSNorm benchmark provider has invalid pre-prepare state");
  }
}

void check_host_oracle() {
  const auto test_case = flagdnn::testing::make_rmsnorm_cases().front();
  const auto plan = flagdnn::validation::ascend::plan_rmsnorm(test_case);
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
  std::vector<float> inverse_variance(static_cast<std::size_t>(plan.rows),
                                      0.0F);
  flagdnn::validation::ascend::rmsnorm_host_oracle(
      plan, input, scale, bias, output, inverse_variance);

  double square_sum = 0.0;
  for (std::size_t column = 0; column < normalized; ++column) {
    square_sum += static_cast<double>(input[column]) * input[column];
  }
  const double expected_inverse =
      1.0 / std::sqrt(square_sum / normalized + plan.epsilon);
  const double expected_output =
      input[0] * expected_inverse * scale[0] + bias[0];
  if (std::abs(inverse_variance[0] - expected_inverse) > 1.0e-6 ||
      std::abs(output[0] - expected_output) > 1.0e-6) {
    throw std::runtime_error("Ascend RMSNorm host oracle formula is invalid");
  }
}

void check_rejections() {
  auto test_case = flagdnn::testing::make_rmsnorm_cases().front();
  test_case.inv_variance.data_type = FLAGDNN_DATA_FLOAT16;
  require_invalid(
      [&] { (void)flagdnn::validation::ascend::plan_rmsnorm(test_case); },
      "RMSNorm non-FP32 inverse variance was accepted");

  test_case = flagdnn::testing::make_rmsnorm_cases().front();
  test_case.scale.dimensions = {5, 17};
  test_case.scale.strides = {17, 1};
  require_invalid(
      [&] { (void)flagdnn::validation::ascend::plan_rmsnorm(test_case); },
      "RMSNorm nonsuffix scale shape was accepted");

  test_case = flagdnn::testing::make_rmsnorm_cases().front();
  test_case.y.uid = test_case.x.uid;
  require_invalid(
      [&] { (void)flagdnn::validation::ascend::plan_rmsnorm(test_case); },
      "RMSNorm aliased binding UID was accepted");

  test_case = flagdnn::testing::make_rmsnorm_cases().front();
  test_case.epsilon = -1.0;
  require_invalid(
      [&] { (void)flagdnn::validation::ascend::plan_rmsnorm(test_case); },
      "RMSNorm negative epsilon was accepted");
}

}  // namespace

int main() {
  if (!flagdnn::testing::aclnn_rmsnorm_status_is_unsupported(
          ACL_ERROR_API_NOT_SUPPORT, "ACLNN RMSNorm is not supported") ||
      flagdnn::testing::aclnn_rmsnorm_status_is_unsupported(
          123456, "invalid RMSNorm descriptor")) {
    throw std::runtime_error(
        "ACLNN RMSNorm unsupported status classification is invalid");
  }
  check_catalog_and_plan();
  check_host_oracle();
  check_rejections();
  return 0;
}
