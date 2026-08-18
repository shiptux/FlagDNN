/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_CONVOLUTION_PROVIDER_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_CONVOLUTION_PROVIDER_HPP_

#include "common/benchmark_provider.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace flagdnn::benchmarking {

struct AclnnConvolutionFpropBenchmarkPlan {
  TensorSpec input;
  TensorSpec filter;
  TensorSpec output;
  TensorSpec convolution_input;
  std::vector<std::int64_t> explicit_padding;
  std::vector<std::int64_t> convolution_padding;
  std::vector<std::int64_t> stride;
  std::vector<std::int64_t> dilation;
  std::int64_t groups = 1;
};

[[nodiscard]] AclnnConvolutionFpropBenchmarkPlan
plan_aclnn_convolution_fprop_benchmark(
    const BenchmarkCase& specification);

[[nodiscard]] std::unique_ptr<BenchmarkExecutable>
build_aclnn_convolution_fprop(const BenchmarkCase& specification);

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_CONVOLUTION_PROVIDER_HPP_
