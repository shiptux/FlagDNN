/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/hygon/context.hpp"

#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include "backends/hygon/error.hpp"

namespace flagdnn::hygon {

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace {

void pop_and_restore_context(hipCtx_t previous_context) noexcept {
  hipCtx_t ignored = nullptr;
  (void)hipCtxPopCurrent(&ignored);

  // Hygon DTK 25.04 leaves the popped context current instead of restoring
  // the prior stack entry. Set only the exact saved context as a compatibility
  // fallback; a trailing hipSetDevice would replace a same-device custom
  // context with that device's primary context.
  hipCtx_t current = nullptr;
  if (hipCtxGetCurrent(&current) != hipSuccess || current != previous_context) {
    (void)hipCtxSetCurrent(previous_context);
  }
}

} // namespace

ContextGuard::ContextGuard(hipDevice_t device, hipCtx_t context) {
  require(context != nullptr, "cannot activate a null HIP context",
          FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
  check_hip(hipCtxGetCurrent(&previous_context_),
            "hipCtxGetCurrent(ContextGuard)");
  if (previous_context_ == context) {
    return;
  }

  check_hip(hipCtxPushCurrent(context), "hipCtxPushCurrent(ContextGuard)");
  active_ = true;
  try {
    hipDevice_t active_device = 0;
    check_hip(hipCtxGetDevice(&active_device), "hipCtxGetDevice(ContextGuard)");
    require(active_device == device,
            "HIP context does not belong to the expected device",
            FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
  } catch (...) {
    pop_and_restore_context(previous_context_);
    active_ = false;
    throw;
  }
}

ContextGuard::~ContextGuard() {
  if (active_) {
    pop_and_restore_context(previous_context_);
  }
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

HygonContext::HygonContext(std::int32_t device_ordinal) {
  if (device_ordinal < 0) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
                     "device ordinal must be nonnegative");
  }
  check_hip(hipInit(0), "hipInit");
  check_hip(hipDeviceGet(&device_, device_ordinal), "hipDeviceGet");
  check_hip(hipDevicePrimaryCtxRetain(&context_, device_),
            "hipDevicePrimaryCtxRetain");
  try {
    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, device_),
              "hipGetDeviceProperties");
    target_fingerprint_ = properties.gcnArchName;
    const std::size_t feature_separator = target_fingerprint_.find(':');
    if (feature_separator != std::string::npos) {
      target_fingerprint_.resize(feature_separator);
    }
    require(target_fingerprint_ == "gfx936",
            "Hygon backend currently supports only gfx936 devices",
            FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);

    hipUUID uuid{};
    check_hip(hipDeviceGetUuid(&uuid, device_), "hipDeviceGetUuid");
    int driver_version = 0;
    check_hip(hipDriverGetVersion(&driver_version), "hipDriverGetVersion");
    std::ostringstream device_identity;
    device_identity << target_fingerprint_ << "-driver" << driver_version
                    << '-';
    for (const char byte : uuid.bytes) {
      device_identity << std::hex << std::setfill('0') << std::setw(2)
                      << static_cast<unsigned int>(
                             static_cast<unsigned char>(byte));
    }
    device_identity_ = device_identity.str();
  } catch (...) {
    (void)hipDevicePrimaryCtxRelease(device_);
    context_ = nullptr;
    throw;
  }
}

HygonContext::~HygonContext() {
  if (context_ != nullptr) {
    (void)hipDevicePrimaryCtxRelease(device_);
  }
}

const std::string &HygonContext::target_fingerprint() const noexcept {
  return target_fingerprint_;
}

EngineBuildContext HygonContext::engine_build_context() const {
  return {device_, context_, target_fingerprint_, device_identity_};
}

} // namespace flagdnn::hygon
