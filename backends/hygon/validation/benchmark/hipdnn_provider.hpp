/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_HIPDNN_PROVIDER_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_HIPDNN_PROVIDER_HPP_

#include "common/benchmark_provider.hpp"
#include "hipdnn_reference.hpp"

#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::benchmarking {

enum class HipdnnReferencePolicy {
  kCorrectnessOracle,
  kPerformance,
};

/*
 * hipDNN convolution support can depend on an actual primitive launch.  This
 * extension keeps that runtime capability probe out of the platform-neutral
 * BenchmarkExecutable contract while allowing the common Hygon runner to use
 * one execution path for every primitive family.
 */
class HipdnnExecutable : public BenchmarkExecutable {
public:
  [[nodiscard]] virtual ProviderCapability
  probe(std::span<const flagdnnBinding_t> bindings, void *workspace,
        std::size_t workspace_size, flagdnnStream_t stream) {
    try {
      execute(bindings, workspace, workspace_size, stream);
    } catch (const validation::hygon::HipdnnStatusError &error) {
      if (!validation::hygon::hipdnn_status_is_capability(error.status())) {
        throw;
      }
      return ProviderCapability::unsupported(
          std::string("hipDNN runtime capability: ") + error.what());
    }
    return {};
  }

  [[nodiscard]] virtual std::string runtime_description() const { return {}; }

  virtual void reject_selected_candidate(std::string reason) {
    (void)reason;
    throw std::logic_error(
        "this hipDNN executable has no selectable candidates");
  }
};

class HipdnnProvider final : public BenchmarkProvider {
public:
  [[nodiscard]] std::string_view name() const noexcept override {
    return "hipdnn";
  }

  [[nodiscard]] ProviderCapability
  capability(const BenchmarkCase &specification) const override;
  [[nodiscard]] ProviderCapability
  capture_capability(const BenchmarkCase &specification) const;
  [[nodiscard]] std::unique_ptr<BenchmarkExecutable>
  build(const BenchmarkCase &specification) override;

  [[nodiscard]] std::unique_ptr<HipdnnExecutable>
  build_reference(const BenchmarkCase &specification,
                  HipdnnReferencePolicy policy) const;
  [[nodiscard]] bool
  requires_accuracy_gated_performance(const BenchmarkCase &specification) const;
  [[nodiscard]] bool uses_fp32_correctness_oracle(
      const BenchmarkCase &specification) const;
  [[nodiscard]] std::vector<TensorSpec>
  reference_specs(const BenchmarkCase &specification,
                  HipdnnReferencePolicy policy) const;
  [[nodiscard]] std::vector<validation::hygon::ReferenceTensor>
  diagnostic_tensors(const BenchmarkCase &specification) const;
  [[nodiscard]] std::string
  operation_name(const BenchmarkCase &specification) const;
  [[nodiscard]] bool
  uses_sequence(const BenchmarkCase &specification) const;
};

} // namespace flagdnn::benchmarking

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_HIPDNN_PROVIDER_HPP_
