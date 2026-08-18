/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_ADD_SQUARE_PROVIDER_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_ADD_SQUARE_PROVIDER_HPP_

#include "common/benchmark_provider.hpp"

namespace flagdnn::benchmarking {

struct AclnnAddSquareBenchmarkPlan {
  TensorSpec left;
  TensorSpec right;
  TensorSpec right_alias;
  TensorSpec square;
  TensorSpec output;
};

[[nodiscard]] AclnnAddSquareBenchmarkPlan plan_aclnn_add_square(
    const BenchmarkCase& specification);

class AclnnAddSquareProvider final : public BenchmarkProvider {
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

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_BENCHMARK_ACLNN_ADD_SQUARE_PROVIDER_HPP_
