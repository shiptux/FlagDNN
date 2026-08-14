/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_TENSOR_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_TENSOR_REFERENCE_HPP_

#include "hipdnn_reference.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::validation::hygon {

enum class HipdnnTensorKind {
  kSlice,
  kReductionAdd,
  kReductionAverage,
  kReductionMultiply,
  kUnavailable,
};

/*
 * Describes only tensor operations with an exact, validated hipDNN primitive
 * mapping.  kUnavailable is intentional: callers use it to report a
 * structured SKIP before constructing a plan.
 */
struct HipdnnTensorOperation {
  HipdnnTensorKind kind = HipdnnTensorKind::kUnavailable;
  std::vector<std::pair<std::int64_t, std::int64_t>> slices;
  std::vector<std::int64_t> slice_strides;
  std::int32_t reduction_axis = 0;
  bool keep_dimensions = false;
  std::string unavailable_reason;
};

[[nodiscard]] HipdnnTensorOperation make_hipdnn_slice_operation(
    std::vector<std::pair<std::int64_t, std::int64_t>> slices,
    std::vector<std::int64_t> strides);

[[nodiscard]] HipdnnTensorOperation
make_hipdnn_reduction_operation(flagdnnReductionMode_t mode, std::int32_t axis,
                                bool keep_dimensions);

[[nodiscard]] HipdnnTensorOperation
make_hipdnn_tensor_unavailable(std::string reason);

/*
 * hipdnnTransformTensor has only been validated here with a dense destination.
 * Runners use this helper for the hipDNN-side output while retaining the
 * original FlagDNN output metadata for the graph execution.
 */
[[nodiscard]] ReferenceTensor
dense_reference_tensor(const ReferenceTensor &tensor);

[[nodiscard]] HipdnnCapability
hipdnn_tensor_capability(const HipdnnTensorOperation &operation,
                         std::span<const ReferenceTensor> tensors);

[[nodiscard]] std::string_view
hipdnn_tensor_kind_name(HipdnnTensorKind kind) noexcept;

class HipdnnTensorPlan final {
public:
  HipdnnTensorPlan(HipdnnTensorOperation operation,
                   std::vector<ReferenceTensor> tensors);
  ~HipdnnTensorPlan();

  HipdnnTensorPlan(const HipdnnTensorPlan &) = delete;
  HipdnnTensorPlan &operator=(const HipdnnTensorPlan &) = delete;
  HipdnnTensorPlan(HipdnnTensorPlan &&) noexcept;
  HipdnnTensorPlan &operator=(HipdnnTensorPlan &&) noexcept;

  [[nodiscard]] std::size_t workspace_size() const noexcept;
  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace flagdnn::validation::hygon

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_TENSOR_REFERENCE_HPP_
