/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/error.hpp"

#include <sstream>
#include <utility>

namespace flagdnn::ascend {

AscendError::AscendError(flagdnnBackendResult_t result, std::string message)
    : std::runtime_error(std::move(message)), result_(result) {}

flagdnnBackendResult_t AscendError::result() const noexcept {
  return result_;
}

std::string acl_error(aclError result, const char* operation) {
  std::ostringstream output;
  output << (operation == nullptr ? "AscendCL operation" : operation)
         << " failed (aclError " << result << ')';
  const char* detail = aclGetRecentErrMsg();
  if (detail != nullptr && detail[0] != '\0') {
    output << ": " << detail;
  }
  return output.str();
}

void check_acl(aclError result, const char* operation) {
  if (result != ACL_SUCCESS) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                      acl_error(result, operation));
  }
}

void require(bool condition,
             const char* message,
             flagdnnBackendResult_t result) {
  if (!condition) {
    throw AscendError(result, message == nullptr ? "invalid value" : message);
  }
}

}  // namespace flagdnn::ascend
