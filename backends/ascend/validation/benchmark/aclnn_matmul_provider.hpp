/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_MATMUL_PROVIDER_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_MATMUL_PROVIDER_HPP_

#include "common/benchmark_provider.hpp"

#include <cstdint>

namespace flagdnn::benchmarking {

struct AclnnMatmulBenchmarkPlan {
  TensorSpec a;
  TensorSpec b;
  TensorSpec output;
  std::int8_t cube_math_type = 0;
};

[[nodiscard]] AclnnMatmulBenchmarkPlan plan_aclnn_matmul_benchmark(
    const BenchmarkCase& specification);

[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_aclnn_matmul(
    const BenchmarkCase& specification);

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_MATMUL_PROVIDER_HPP_
