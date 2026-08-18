/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_CONTEXT_HPP_
#define FLAGDNN_BACKENDS_HYGON_CONTEXT_HPP_

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <string>

namespace flagdnn::hygon {

struct EngineBuildContext {
  hipDevice_t device = 0;
  hipCtx_t context = nullptr;
  std::string target_fingerprint;
  std::string device_identity;
};

class ContextGuard {
public:
  ContextGuard(hipDevice_t device, hipCtx_t context);
  ~ContextGuard();

  ContextGuard(const ContextGuard &) = delete;
  ContextGuard &operator=(const ContextGuard &) = delete;

private:
  hipCtx_t previous_context_ = nullptr;
  bool active_ = false;
};

class HygonContext {
public:
  explicit HygonContext(std::int32_t device_ordinal);
  ~HygonContext();

  HygonContext(const HygonContext &) = delete;
  HygonContext &operator=(const HygonContext &) = delete;

  [[nodiscard]] const std::string &target_fingerprint() const noexcept;
  [[nodiscard]] EngineBuildContext engine_build_context() const;

private:
  hipDevice_t device_ = 0;
  hipCtx_t context_ = nullptr;
  std::string target_fingerprint_;
  std::string device_identity_;
};

} // namespace flagdnn::hygon

#endif // FLAGDNN_BACKENDS_HYGON_CONTEXT_HPP_
