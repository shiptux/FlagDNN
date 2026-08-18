/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_BATCHNORM_INFERENCE_VALIDATION_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_BATCHNORM_INFERENCE_VALIDATION_HPP_

#include "common/normalization.hpp"
#include "common/case.hpp"

#include <span>
#include <vector>

namespace flagdnn::validation::ascend {

struct BatchnormInferencePlan {
  testing::TestTensor input;
  testing::TestTensor weight;
  testing::TestTensor bias;
  testing::TestTensor mean;
  testing::TestTensor inverse_standard_deviation;
  testing::TestTensor output;
};

[[nodiscard]] BatchnormInferencePlan plan_batchnorm_inference(
    const testing::BatchnormInferenceTestCase& test_case);
[[nodiscard]] BatchnormInferencePlan plan_batchnorm_inference(
    const benchmarking::BenchmarkCase& test_case);
[[nodiscard]] testing::TestTensor batchnorm_inference_reference_data_tensor(
    const testing::TestTensor& tensor);
[[nodiscard]] benchmarking::BenchmarkCase
batchnorm_inference_reference_benchmark_case(
    const benchmarking::BenchmarkCase& test_case);

[[nodiscard]] std::vector<testing::BatchnormInferenceTestCase>
make_ascend_batchnorm_inference_cases(
    std::span<const testing::BatchnormInferenceTestCase> common_cases);
[[nodiscard]] std::vector<benchmarking::BenchmarkCase>
make_ascend_batchnorm_inference_benchmark_cases(
    std::span<const benchmarking::BenchmarkCase> common_cases);

void batchnorm_inference_host_oracle(
    const BatchnormInferencePlan& plan,
    std::span<const float> input_storage,
    std::span<const float> mean,
    std::span<const float> inverse_standard_deviation,
    std::span<const float> scale,
    std::span<const float> bias,
    std::span<float> output_storage);

}  // namespace flagdnn::validation::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_BATCHNORM_INFERENCE_VALIDATION_HPP_
