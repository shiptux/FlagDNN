/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_HIPDNN_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_HIPDNN_REFERENCE_HPP_

#include <flagdnn/flagdnn.h>
#include <hipdnn.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::validation::hygon {

struct ReferenceTensor {
  std::int64_t uid = 0;
  flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT32;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  std::size_t binding_byte_offset = 0;
};

template <typename Tensor>
ReferenceTensor as_reference_tensor(const Tensor &tensor) {
  return {tensor.uid, tensor.data_type, tensor.dimensions, tensor.strides,
          tensor.binding_byte_offset};
}

enum class HipdnnCapabilityClass {
  kSupported,
  kVendorUnsupported,
  kInvalidAdapterContract,
};

struct HipdnnCapability {
  bool supported = true;
  HipdnnCapabilityClass classification = HipdnnCapabilityClass::kSupported;
  std::string reason;

  [[nodiscard]] static HipdnnCapability vendor_unsupported(std::string reason) {
    return {false, HipdnnCapabilityClass::kVendorUnsupported,
            std::move(reason)};
  }

  [[nodiscard]] static HipdnnCapability
  invalid_adapter_contract(std::string reason) {
    return {false, HipdnnCapabilityClass::kInvalidAdapterContract,
            std::move(reason)};
  }
};

void require_valid_hipdnn_adapter_contract(const HipdnnCapability &capability,
                                           std::string_view operation);

/*
 * Preserve the native hipDNN status across adapter layers. Validation may
 * turn only the explicit capability statuses into a structured SKIP;
 * malformed descriptors, allocation failures, execution failures and all
 * other errors remain hard failures.
 */
[[nodiscard]] std::string_view
hipdnn_status_name(hipdnnStatus_t status) noexcept;

[[nodiscard]] bool hipdnn_status_is_capability(hipdnnStatus_t status) noexcept;

class HipdnnStatusError final : public std::runtime_error {
public:
  HipdnnStatusError(hipdnnStatus_t status, std::string_view operation);

  [[nodiscard]] hipdnnStatus_t status() const noexcept { return status_; }

private:
  hipdnnStatus_t status_;
};

void check_hipdnn_status(hipdnnStatus_t status, std::string_view operation);

[[nodiscard]] std::string hipdnn_environment();
[[nodiscard]] std::string
describe_reference_tensors(std::span<const ReferenceTensor> tensors);

} // namespace flagdnn::validation::hygon

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_HIPDNN_REFERENCE_HPP_
