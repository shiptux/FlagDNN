/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_LAYOUT_PROVIDER_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_LAYOUT_PROVIDER_HPP_

#include "common/benchmark_provider.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace flagdnn::benchmarking {

struct AclnnLayoutBenchmarkPlan {
  TensorSpec input;
  TensorSpec output;
};

class AclnnLayoutBenchmarkUnsupportedError final
    : public std::runtime_error {
 public:
  AclnnLayoutBenchmarkUnsupportedError(std::int32_t status,
                                       std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

[[nodiscard]] AclnnLayoutBenchmarkPlan plan_aclnn_layout_benchmark(
    const BenchmarkCase& specification);

class AclnnLayoutProvider final : public BenchmarkProvider {
 public:
  [[nodiscard]] std::string_view name() const noexcept override {
    return "aclnn";
  }
  [[nodiscard]] ProviderCapability capability(
      const BenchmarkCase& specification) const override;
  [[nodiscard]] std::unique_ptr<BenchmarkExecutable> build(
      const BenchmarkCase& specification) override;
};

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_LAYOUT_PROVIDER_HPP_
