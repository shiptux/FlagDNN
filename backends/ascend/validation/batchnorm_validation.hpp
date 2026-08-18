/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_BATCHNORM_VALIDATION_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_BATCHNORM_VALIDATION_HPP_

#include "common/case.hpp"
#include "common/normalization.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace flagdnn::validation::ascend {

struct BatchnormPlan {
  testing::TestTensor input;
  testing::TestTensor scale;
  testing::TestTensor bias;
  testing::TestTensor previous_running_mean;
  testing::TestTensor previous_running_variance;
  testing::TestTensor output;
  testing::TestTensor mean;
  testing::TestTensor inverse_standard_deviation;
  testing::TestTensor next_running_mean;
  testing::TestTensor next_running_variance;
  double epsilon = 0.0;
  double momentum = 0.0;
  std::size_t batch = 0;
  std::size_t channels = 0;
  std::size_t spatial = 0;
  std::size_t reduction_elements = 0;
};

[[nodiscard]] BatchnormPlan plan_batchnorm(
    const testing::BatchnormTestCase& test_case);
[[nodiscard]] BatchnormPlan plan_batchnorm(
    const benchmarking::BenchmarkCase& test_case);
[[nodiscard]] testing::TestTensor batchnorm_reference_data_tensor(
    const testing::TestTensor& tensor);
[[nodiscard]] benchmarking::BenchmarkCase batchnorm_reference_benchmark_case(
    const benchmarking::BenchmarkCase& test_case);

[[nodiscard]] std::vector<testing::BatchnormTestCase>
make_ascend_batchnorm_cases(
    std::span<const testing::BatchnormTestCase> common_cases);
[[nodiscard]] std::vector<benchmarking::BenchmarkCase>
make_ascend_batchnorm_benchmark_cases(
    std::span<const benchmarking::BenchmarkCase> common_cases);

void batchnorm_reference_variance_to_invstd(
    std::span<float> saved_variance, double epsilon);

void batchnorm_host_oracle(
    const BatchnormPlan& plan,
    std::span<const float> input_storage,
    std::span<const float> scale,
    std::span<const float> bias,
    std::span<const float> previous_running_mean,
    std::span<const float> previous_running_variance,
    std::span<float> output_storage,
    std::span<float> mean,
    std::span<float> inverse_standard_deviation,
    std::span<float> next_running_mean,
    std::span<float> next_running_variance);

}  // namespace flagdnn::validation::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_BATCHNORM_VALIDATION_HPP_
