/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_RMSNORM_PROVIDER_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_RMSNORM_PROVIDER_HPP_

#include "common/benchmark_provider.hpp"

#include <memory>

namespace flagdnn::benchmarking {

[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_aclnn_rmsnorm(
    const BenchmarkCase& specification);

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_RMSNORM_PROVIDER_HPP_
