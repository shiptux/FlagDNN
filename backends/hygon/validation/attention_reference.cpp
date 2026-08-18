/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "attention_reference.hpp"

#include <span>
#include <string_view>

namespace flagdnn::validation::hygon {

HipdnnCapability
hipdnn_attention_capability(const HipdnnAttentionOperation &operation,
                            std::span<const ReferenceTensor> tensors) {
  static_cast<void>(tensors);
  switch (operation.kind) {
  case HipdnnAttentionKind::kSdpa:
    return HipdnnCapability::vendor_unsupported(
        "public hipDNN primitive unavailable: API has no "
        "MatMul/SDPA/Graph primitive for an exact SDPA forward reference");
  case HipdnnAttentionKind::kSdpaBackward:
    return HipdnnCapability::vendor_unsupported(
        "public hipDNN primitive unavailable: API has no "
        "MatMul/SDPA/Graph primitive for an exact SDPA backward reference");
  case HipdnnAttentionKind::kSdpaFp8:
    return HipdnnCapability::vendor_unsupported(
        "public hipDNN primitive unavailable: API has no FP8 "
        "MatMul/SDPA/Graph primitive for an exact FP8 SDPA forward reference");
  case HipdnnAttentionKind::kSdpaFp8Backward:
    return HipdnnCapability::vendor_unsupported(
        "public hipDNN primitive unavailable: API has no FP8 "
        "MatMul/SDPA/Graph primitive for an exact FP8 SDPA backward reference");
  }
  return HipdnnCapability::invalid_adapter_contract(
      "unknown hipDNN attention operation");
}

std::string_view hipdnn_attention_kind_name(HipdnnAttentionKind kind) noexcept {
  switch (kind) {
  case HipdnnAttentionKind::kSdpa:
    return "sdpa";
  case HipdnnAttentionKind::kSdpaBackward:
    return "sdpa_backward";
  case HipdnnAttentionKind::kSdpaFp8:
    return "sdpa_fp8";
  case HipdnnAttentionKind::kSdpaFp8Backward:
    return "sdpa_fp8_backward";
  }
  return "attention_unknown";
}

} // namespace flagdnn::validation::hygon
