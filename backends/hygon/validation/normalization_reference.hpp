/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_NORMALIZATION_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_NORMALIZATION_REFERENCE_HPP_

#include "hipdnn_reference.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::validation::hygon {

enum class HipdnnNormalizationKind {
  kBatchnormTraining,
  kUnavailable,
};

/*
 * The tensor order for kBatchnormTraining is deliberately identical to the
 * common FlagDNN case contract:
 *   X, scale, bias, previous running mean, previous running variance,
 *   Y, saved mean, saved inverse variance, next running mean,
 *   next running variance.
 */
struct HipdnnNormalizationOperation {
  HipdnnNormalizationKind kind = HipdnnNormalizationKind::kUnavailable;
  double epsilon = 0.0;
  double momentum = 0.0;
  std::string unavailable_reason;
};

[[nodiscard]] HipdnnNormalizationOperation
make_hipdnn_batchnorm_training_operation(double epsilon, double momentum);

[[nodiscard]] HipdnnNormalizationOperation
make_hipdnn_normalization_unavailable(std::string reason);

[[nodiscard]] HipdnnCapability
hipdnn_normalization_capability(const HipdnnNormalizationOperation &operation,
                                std::span<const ReferenceTensor> tensors);

[[nodiscard]] std::string_view
hipdnn_normalization_kind_name(HipdnnNormalizationKind kind) noexcept;

/* hipDNN BatchNorm is validated with dense NCHW X/Y reference storage. */
[[nodiscard]] ReferenceTensor
batchnorm_dense_reference_tensor(const ReferenceTensor &tensor);

class HipdnnNormalizationPlan final {
public:
  HipdnnNormalizationPlan(HipdnnNormalizationOperation operation,
                          std::vector<ReferenceTensor> tensors);
  ~HipdnnNormalizationPlan();

  HipdnnNormalizationPlan(const HipdnnNormalizationPlan &) = delete;
  HipdnnNormalizationPlan &operator=(const HipdnnNormalizationPlan &) = delete;
  HipdnnNormalizationPlan(HipdnnNormalizationPlan &&) noexcept;
  HipdnnNormalizationPlan &operator=(HipdnnNormalizationPlan &&) noexcept;

  [[nodiscard]] std::size_t workspace_size() const noexcept;
  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace flagdnn::validation::hygon

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_NORMALIZATION_REFERENCE_HPP_
