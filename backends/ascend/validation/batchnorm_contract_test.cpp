/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/batchnorm_validation.hpp"
#include "validation/benchmark/aclnn_batchnorm_provider.hpp"
#include "validation/functional/aclnn_batchnorm.hpp"
#include "validation/tensor_io.hpp"

#include "common/cases.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace acl = flagdnn::validation::ascend;
namespace io = flagdnn::validation::ascend::tensor_io;
using flagdnn::testing::BatchnormTestCase;
using flagdnn::testing::TestTensor;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_close(float actual, float expected, float tolerance,
                   const std::string& message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(message);
  }
}

void require_failure(const std::function<void()>& action,
                     const std::string& message) {
  try {
    action();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

TestTensor tensor(std::int64_t uid,
                  flagdnnDataType_t data_type,
                  std::vector<std::int64_t> dimensions,
                  std::vector<std::int64_t> strides) {
  return {uid, data_type, std::move(dimensions), std::move(strides), 0};
}

BatchnormTestCase oracle_case() {
  BatchnormTestCase result;
  result.name = "batchnorm_validation_oracle";
  result.x = tensor(1, FLAGDNN_DATA_FLOAT32, {2, 2, 2, 1}, {12, 4, 2, 1});
  result.scale = tensor(2, FLAGDNN_DATA_FLOAT32, {1, 2, 1, 1}, {2, 1, 1, 1});
  result.bias = tensor(3, FLAGDNN_DATA_FLOAT32, {1, 2, 1, 1}, {2, 1, 1, 1});
  result.previous_running_mean =
      tensor(4, FLAGDNN_DATA_FLOAT32, {1, 2, 1, 1}, {2, 1, 1, 1});
  result.previous_running_variance =
      tensor(5, FLAGDNN_DATA_FLOAT32, {1, 2, 1, 1}, {2, 1, 1, 1});
  result.y = tensor(6, FLAGDNN_DATA_FLOAT32, {2, 2, 2, 1}, {14, 5, 2, 1});
  result.mean = tensor(7, FLAGDNN_DATA_FLOAT32, {1, 2, 1, 1}, {2, 1, 1, 1});
  result.inv_variance =
      tensor(8, FLAGDNN_DATA_FLOAT32, {1, 2, 1, 1}, {2, 1, 1, 1});
  result.next_running_mean =
      tensor(9, FLAGDNN_DATA_FLOAT32, {1, 2, 1, 1}, {2, 1, 1, 1});
  result.next_running_variance =
      tensor(10, FLAGDNN_DATA_FLOAT32, {1, 2, 1, 1}, {2, 1, 1, 1});
  result.epsilon = 0.25;
  result.momentum = 0.25;
  result.absolute_tolerance = 1.0e-5;
  result.relative_tolerance = 1.0e-5;
  return result;
}

void test_catalog_and_reference_contract() {
  const auto common = flagdnn::testing::make_batchnorm_cases();
  const auto functional = acl::make_ascend_batchnorm_cases(common);
  require(functional.size() == 6, "BatchNorm functional catalog is not exact");
  for (const auto& test_case : functional) {
    const auto plan = acl::plan_batchnorm(test_case);
    require(plan.batch == 2 && plan.channels == 8 && plan.spatial == 64 &&
                plan.reduction_elements == 128,
            "BatchNorm functional plan dimensions are invalid");
  }

  const auto common_benchmark =
      flagdnn::benchmarking::batchnorm_benchmark_cases();
  const auto benchmark =
      acl::make_ascend_batchnorm_benchmark_cases(common_benchmark);
  require(benchmark.size() == 24,
          "BatchNorm benchmark catalog is not exact");
  const auto benchmark_plan = acl::plan_batchnorm(benchmark.front());
  require(benchmark_plan.channels == 32,
          "BatchNorm benchmark plan channels are invalid");

  const auto& channels_last = functional.at(3);
  const auto reference = acl::batchnorm_reference_data_tensor(channels_last.x);
  require(reference.dimensions == channels_last.x.dimensions &&
              reference.strides == std::vector<std::int64_t>({512, 64, 8, 1}) &&
              reference.binding_byte_offset == 0,
          "BatchNorm ACLNN reference data layout is invalid");
  const auto reference_benchmark =
      acl::batchnorm_reference_benchmark_case(benchmark.front());
  require(reference_benchmark.tensors.front().strides ==
              flagdnn::benchmarking::contiguous_strides(
                  reference_benchmark.tensors.front().dimensions) &&
              reference_benchmark.tensors.at(5).strides ==
                  flagdnn::benchmarking::contiguous_strides(
                      reference_benchmark.tensors.at(5).dimensions),
          "BatchNorm benchmark reference layout is invalid");

  const auto reference_plan =
      flagdnn::testing::plan_aclnn_batchnorm(functional.front());
  require(reference_plan.training && reference_plan.binding_count == 10 &&
              reference_plan.output_count == 5 &&
              reference_plan.operation.channels == 8,
          "ACLNN BatchNorm training reference plan is invalid");
  const auto reference_executable =
      flagdnn::testing::build_batchnorm_reference(functional.front());
  require(reference_executable != nullptr &&
              reference_executable->workspace_size() == 0,
          "ACLNN BatchNorm reference pre-prepare state is invalid");
  const auto benchmark_executable =
      flagdnn::benchmarking::build_aclnn_batchnorm(benchmark.front());
  require(benchmark_executable != nullptr &&
              benchmark_executable->workspace_size() == 0,
          "ACLNN BatchNorm benchmark pre-prepare state is invalid");
}

void test_host_oracle() {
  const BatchnormTestCase test_case = oracle_case();
  const acl::BatchnormPlan plan = acl::plan_batchnorm(test_case);
  std::vector<float> input(io::storage_element_count(plan.input),
                           io::kPaddingSentinel);
  const std::vector<float> logical = {1, 3, 2, 4, 5, 7, 6, 8};
  for (std::size_t index = 0; index < logical.size(); ++index) {
    input[io::physical_offset(index, plan.input)] = logical[index];
  }
  std::vector<float> output(io::storage_element_count(plan.output),
                            io::kPaddingSentinel);
  const std::vector<float> scale = {2.0F, -1.0F};
  const std::vector<float> bias = {1.0F, 0.5F};
  const std::vector<float> previous_mean = {10.0F, 20.0F};
  const std::vector<float> previous_variance = {30.0F, 40.0F};
  std::vector<float> mean(2), invstd(2), next_mean(2), next_variance(2);
  acl::batchnorm_host_oracle(plan,
                             input,
                             scale,
                             bias,
                             previous_mean,
                             previous_variance,
                             output,
                             mean,
                             invstd,
                             next_mean,
                             next_variance);

  const float expected_invstd = 1.0F / std::sqrt(5.25F);
  require_close(mean[0], 4.0F, 1.0e-6F, "BatchNorm mean[0] is invalid");
  require_close(mean[1], 5.0F, 1.0e-6F, "BatchNorm mean[1] is invalid");
  require_close(invstd[0], expected_invstd, 1.0e-6F,
                "BatchNorm invstd is invalid");
  require_close(next_mean[0], 8.5F, 1.0e-6F,
                "BatchNorm next running mean is invalid");
  require_close(next_variance[0], 24.166666F, 1.0e-5F,
                "BatchNorm next running variance is invalid");
  std::vector<float> aclnn_saved_variance = {5.0F, 8.75F};
  acl::batchnorm_reference_variance_to_invstd(
      aclnn_saved_variance, test_case.epsilon);
  require_close(aclnn_saved_variance[0], expected_invstd, 1.0e-6F,
                "ACLNN BatchNorm saved variance conversion is invalid");
  for (std::size_t index = 0; index < logical.size(); ++index) {
    const std::size_t channel = (index / 2) % 2;
    const float expected = (logical[index] - mean[channel]) *
                               expected_invstd * scale[channel] +
                           bias[channel];
    require_close(output[io::physical_offset(index, plan.output)],
                  expected,
                  1.0e-5F,
                  "BatchNorm logical output is invalid");
  }
  io::require_padding_unchanged("BatchNorm", output, plan.output);
}

void test_invalid_contracts() {
  BatchnormTestCase invalid = oracle_case();
  invalid.scale.uid = invalid.x.uid;
  require_failure([&] { (void)acl::plan_batchnorm(invalid); },
                  "BatchNorm duplicate UID was accepted");

  invalid = oracle_case();
  invalid.scale.data_type = FLAGDNN_DATA_BFLOAT16;
  require_failure([&] { (void)acl::plan_batchnorm(invalid); },
                  "BatchNorm mixed parameter dtype was accepted");

  invalid = oracle_case();
  invalid.mean.dimensions = {3};
  invalid.mean.strides = {1};
  require_failure([&] { (void)acl::plan_batchnorm(invalid); },
                  "BatchNorm invalid statistic size was accepted");

  invalid = oracle_case();
  invalid.momentum = 1.25;
  require_failure([&] { (void)acl::plan_batchnorm(invalid); },
                  "BatchNorm invalid momentum was accepted");

  invalid = oracle_case();
  invalid.y.dimensions[0] = 3;
  require_failure([&] { (void)acl::plan_batchnorm(invalid); },
                  "BatchNorm mismatched output shape was accepted");
}

}  // namespace

int main() {
  try {
    require(flagdnn::testing::aclnn_batchnorm_status_is_unsupported(
                ACL_ERROR_API_NOT_SUPPORT,
                "ACLNN BatchNorm is not supported"),
            "known ACLNN BatchNorm unsupported status was not classified");
    require(!flagdnn::testing::aclnn_batchnorm_status_is_unsupported(
                123456,
                "invalid BatchNorm descriptor"),
            "unexpected BatchNorm error was classified as unsupported");
    test_catalog_and_reference_contract();
    test_host_oracle();
    test_invalid_contracts();
    std::cout << "BatchNorm validation contract passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
