/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_PROVIDER_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_PROVIDER_HPP_

#include "common/benchmark_provider.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace flagdnn::benchmarking {

struct AclnnBinaryPointwiseBenchmarkPlan {
  TensorSpec left;
  TensorSpec right;
  TensorSpec output;
  flagdnnPointwiseMode_t mode = FLAGDNN_POINTWISE_NOT_SET;
  double alpha = 1.0;
  bool uses_contiguous_reference_output = false;
};

struct AclnnTernaryPointwiseBenchmarkPlan {
  TensorSpec self;
  TensorSpec other;
  TensorSpec condition;
  TensorSpec output;
  flagdnnPointwiseMode_t mode = FLAGDNN_POINTWISE_NOT_SET;
};

struct AclnnUnaryPointwiseBenchmarkPlan {
  TensorSpec input;
  TensorSpec output;
  flagdnnPointwiseMode_t mode = FLAGDNN_POINTWISE_NOT_SET;
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  bool uses_contiguous_reference_output = false;
};

struct AclnnReductionBenchmarkPlan {
  TensorSpec input;
  TensorSpec output;
  flagdnnReductionMode_t mode = FLAGDNN_REDUCTION_ADD;
  std::int64_t axis = 0;
  bool keep_dimensions = false;
};

class AclnnBenchmarkUnsupportedError final : public std::runtime_error {
 public:
  AclnnBenchmarkUnsupportedError(std::int32_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

[[nodiscard]] AclnnBinaryPointwiseBenchmarkPlan
plan_aclnn_binary_pointwise(
    const BenchmarkCase& specification);

[[nodiscard]] AclnnTernaryPointwiseBenchmarkPlan
plan_aclnn_ternary_pointwise(
    const BenchmarkCase& specification);

[[nodiscard]] bool is_aclnn_unary_benchmark(
    const BenchmarkCase& specification) noexcept;

[[nodiscard]] AclnnUnaryPointwiseBenchmarkPlan
plan_aclnn_unary_pointwise(const BenchmarkCase& specification);

[[nodiscard]] std::unique_ptr<BenchmarkExecutable>
build_aclnn_unary_pointwise(const BenchmarkCase& specification);

[[nodiscard]] AclnnReductionBenchmarkPlan plan_aclnn_reduction(
    const BenchmarkCase& specification);

[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_aclnn_reduction(
    const BenchmarkCase& specification);

class AclnnProvider final : public BenchmarkProvider {
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

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_PROVIDER_HPP_
