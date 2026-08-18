/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_RMSNORM_VALIDATION_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_RMSNORM_VALIDATION_HPP_

#include "common/case.hpp"
#include "common/normalization.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace flagdnn::validation::ascend {

struct RmsnormPlan {
  testing::TestTensor input;
  testing::TestTensor scale;
  testing::TestTensor bias;
  testing::TestTensor output;
  testing::TestTensor inverse_variance;
  std::int64_t rows = 0;
  std::int64_t normalized_elements = 0;
  double epsilon = 0.0;
};

[[nodiscard]] RmsnormPlan plan_rmsnorm(
    const testing::RmsnormTestCase& test_case);
[[nodiscard]] RmsnormPlan plan_rmsnorm(
    const benchmarking::BenchmarkCase& test_case);

[[nodiscard]] std::vector<testing::RmsnormTestCase>
make_ascend_rmsnorm_cases(
    std::span<const testing::RmsnormTestCase> common_cases);
[[nodiscard]] std::vector<benchmarking::BenchmarkCase>
make_ascend_rmsnorm_benchmark_cases(
    std::span<const benchmarking::BenchmarkCase> common_cases);

void rmsnorm_host_oracle(const RmsnormPlan& plan,
                         std::span<const float> input_storage,
                         std::span<const float> scale,
                         std::span<const float> bias,
                         std::span<float> output_storage,
                         std::span<float> inverse_variance_storage);

}  // namespace flagdnn::validation::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_RMSNORM_VALIDATION_HPP_
