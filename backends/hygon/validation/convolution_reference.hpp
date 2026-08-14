/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_CONVOLUTION_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_CONVOLUTION_REFERENCE_HPP_

#include "hipdnn_reference.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::validation::hygon {

enum class HipdnnConvolutionKind {
  kFprop,
  kDgrad,
  kWgrad,
  kConvBiasRelu,
  kUnavailable,
};

enum class HipdnnConvolutionMode {
  kCrossCorrelation,
  kConvolution,
};

enum class HipdnnConvolutionAlgorithmPolicy {
  kCorrectnessOracle,
  kPerformance,
};

/*
 * Tensors always use semantic X/W/Y order.  ConvBiasRelu additionally places
 * bias after Y, yielding X/W/Y/bias.  Keeping the semantic order independent
 * of the public operation's input/output order makes the three primitive
 * calls share one descriptor implementation.
 */
struct HipdnnConvolutionOperation {
  HipdnnConvolutionKind kind = HipdnnConvolutionKind::kUnavailable;
  std::vector<std::int64_t> padding;
  std::vector<std::int64_t> stride;
  std::vector<std::int64_t> dilation;
  std::int64_t groups = 1;
  HipdnnConvolutionMode mode = HipdnnConvolutionMode::kCrossCorrelation;
  std::string unavailable_reason;
};

[[nodiscard]] HipdnnConvolutionOperation make_hipdnn_convolution_operation(
    HipdnnConvolutionKind kind, std::vector<std::int64_t> pre_padding,
    std::vector<std::int64_t> post_padding, std::vector<std::int64_t> stride,
    std::vector<std::int64_t> dilation, std::int64_t groups,
    HipdnnConvolutionMode mode = HipdnnConvolutionMode::kCrossCorrelation);

[[nodiscard]] HipdnnConvolutionOperation
make_hipdnn_convolution_unavailable(std::string reason);

/* Structural gate only.  Runners must additionally call probe_execute(). */
[[nodiscard]] HipdnnCapability
hipdnn_convolution_capability(const HipdnnConvolutionOperation &operation,
                              std::span<const ReferenceTensor> tensors);

[[nodiscard]] std::string_view
hipdnn_convolution_kind_name(HipdnnConvolutionKind kind) noexcept;

[[nodiscard]] std::string_view hipdnn_convolution_algorithm_policy_name(
    HipdnnConvolutionAlgorithmPolicy policy) noexcept;

/*
 * Pure policy contract used by validation tests and by the descriptor plan.
 * Heuristic values only influence performance ordering.  Correctness always
 * uses one fixed, deterministic public hipDNN primitive algorithm; the same
 * oracle is retained as the final fallback in the performance order.
 */
[[nodiscard]] std::vector<int>
hipdnn_convolution_algorithm_order(HipdnnConvolutionKind kind,
                                   HipdnnConvolutionAlgorithmPolicy policy,
                                   std::span<const int> heuristic_values);

[[nodiscard]] std::string_view
hipdnn_convolution_algorithm_name(HipdnnConvolutionKind kind,
                                  int algorithm) noexcept;

/*
 * The v7 algorithm query is only an ordering hint.  At this metadata-only
 * layer, NOT_SUPPORTED and EXECUTION_FAILED permit exhaustive public-enum
 * fallback; every other query failure is a broken validation path.  Actual
 * primitive execution retains the stricter NOT_SUPPORTED-only capability
 * classification.
 */
[[nodiscard]] bool hipdnn_convolution_heuristic_allows_enum_fallback(
    hipdnnStatus_t status) noexcept;

class HipdnnConvolutionPlan final {
public:
  HipdnnConvolutionPlan(HipdnnConvolutionOperation operation,
                        std::vector<ReferenceTensor> tensors,
                        HipdnnConvolutionAlgorithmPolicy policy);
  ~HipdnnConvolutionPlan();

  HipdnnConvolutionPlan(const HipdnnConvolutionPlan &) = delete;
  HipdnnConvolutionPlan &operator=(const HipdnnConvolutionPlan &) = delete;
  HipdnnConvolutionPlan(HipdnnConvolutionPlan &&) noexcept;
  HipdnnConvolutionPlan &operator=(HipdnnConvolutionPlan &&) noexcept;

  /* Descriptor/algo-query capability.  No device execution has occurred yet. */
  [[nodiscard]] HipdnnCapability capability() const;
  [[nodiscard]] std::size_t workspace_size() const noexcept;
  [[nodiscard]] HipdnnConvolutionAlgorithmPolicy policy() const noexcept;
  [[nodiscard]] std::optional<int> selected_algorithm() const noexcept;
  [[nodiscard]] std::string_view selected_algorithm_name() const noexcept;

  /*
   * Executes the real public hipDNN primitive and synchronizes its stream.
   * On success the working algorithm is retained for steady-state execute().
   */
  [[nodiscard]] HipdnnCapability
  probe_execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
                std::size_t workspace_size, flagdnnStream_t stream);

  /* Continue performance selection after an executed candidate failed the
   * numerical comparison against the correctness oracle.  Rejecting the last
   * candidate makes the next probe report exhaustion; the benchmark runner
   * owns the dtype-specific hard-fail versus structured-SKIP policy. */
  void reject_selected_algorithm(std::string reason);

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace flagdnn::validation::hygon

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_CONVOLUTION_REFERENCE_HPP_
