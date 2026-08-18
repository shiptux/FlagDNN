/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_ATTENTION_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_ATTENTION_REFERENCE_HPP_

#include "hipdnn_reference.hpp"

#include <span>
#include <string_view>

namespace flagdnn::validation::hygon {

enum class HipdnnAttentionKind {
  kSdpa,
  kSdpaBackward,
  kSdpaFp8,
  kSdpaFp8Backward,
};

/*
 * Describes the exact public hipDNN primitive reference boundary.  The
 * installed primitive API has no MatMul, SDPA, or Graph entry point, so every
 * current attention kind is intentionally reported as unavailable.  This
 * object still centralizes that capability decision so runners cannot fall
 * back to a host oracle or another vendor library.
 */
struct HipdnnAttentionOperation {
  HipdnnAttentionKind kind = HipdnnAttentionKind::kSdpa;
};

[[nodiscard]] HipdnnCapability
hipdnn_attention_capability(const HipdnnAttentionOperation &operation,
                            std::span<const ReferenceTensor> tensors);

[[nodiscard]] std::string_view
hipdnn_attention_kind_name(HipdnnAttentionKind kind) noexcept;

} // namespace flagdnn::validation::hygon

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_ATTENTION_REFERENCE_HPP_
