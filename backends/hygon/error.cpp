/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/hygon/error.hpp"

#include <sstream>
#include <utility>

namespace flagdnn::hygon {

HygonError::HygonError(flagdnnBackendResult_t result, std::string message)
    : std::runtime_error(std::move(message)), result_(result) {}

HygonError::HygonError(flagdnnBackendResult_t result, std::string message,
                       hipError_t hip_result)
    : std::runtime_error(std::move(message)), result_(result),
      hip_result_(hip_result) {}

flagdnnBackendResult_t HygonError::result() const noexcept { return result_; }

std::optional<hipError_t> HygonError::hip_result() const noexcept {
  return hip_result_;
}

std::string hip_error(hipError_t result, const char *operation) {
  const char *name = hipGetErrorName(result);
  const char *description = hipGetErrorString(result);
  std::ostringstream output;
  output << operation << " failed";
  if (name != nullptr) {
    output << " (" << name << ')';
  }
  if (description != nullptr) {
    output << ": " << description;
  }
  return output.str();
}

void check_hip(hipError_t result, const char *operation) {
  if (result != hipSuccess) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                     hip_error(result, operation), result);
  }
}

void require(bool condition, const char *message,
             flagdnnBackendResult_t result) {
  if (!condition) {
    throw HygonError(result, message);
  }
}

} // namespace flagdnn::hygon
