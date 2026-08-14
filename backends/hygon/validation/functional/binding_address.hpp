/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_BINDING_ADDRESS_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_BINDING_ADDRESS_HPP_

#include <cstddef>
#include <cstdint>

namespace flagdnn::testing::hygon_functional {

enum class BindingAddress { kStorageBase, kTensorEntrance };

/*
 * hipDNN plans add ReferenceTensor::binding_byte_offset themselves, so their
 * binding is the storage base. FlagDNN's public binding contract is already
 * the tensor entrance and therefore receives base + offset.
 */
inline void *binding_pointer(void *storage_base, std::size_t byte_offset,
                             BindingAddress address) noexcept {
  if (address == BindingAddress::kStorageBase) {
    return storage_base;
  }
  return static_cast<void *>(static_cast<std::uint8_t *>(storage_base) +
                             byte_offset);
}

} // namespace flagdnn::testing::hygon_functional

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_BINDING_ADDRESS_HPP_
