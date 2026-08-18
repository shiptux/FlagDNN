/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_ERROR_HPP_
#define FLAGDNN_BACKENDS_HYGON_ERROR_HPP_

#include <hip/hip_runtime_api.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "backends/backend_api.h"

namespace flagdnn::hygon {

class HygonError : public std::runtime_error {
public:
  HygonError(flagdnnBackendResult_t result, std::string message);
  HygonError(flagdnnBackendResult_t result, std::string message,
             hipError_t hip_result);

  [[nodiscard]] flagdnnBackendResult_t result() const noexcept;
  [[nodiscard]] std::optional<hipError_t> hip_result() const noexcept;

private:
  flagdnnBackendResult_t result_;
  std::optional<hipError_t> hip_result_;
};

[[nodiscard]] std::string hip_error(hipError_t result, const char *operation);
void check_hip(hipError_t result, const char *operation);
void require(
    bool condition, const char *message,
    flagdnnBackendResult_t result = FLAGDNN_BACKEND_RESULT_INVALID_VALUE);

} // namespace flagdnn::hygon

#endif // FLAGDNN_BACKENDS_HYGON_ERROR_HPP_
