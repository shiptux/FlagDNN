/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ERROR_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ERROR_HPP_

#include "backends/backend_api.h"

#include <acl/acl_rt.h>

#include <stdexcept>
#include <string>

namespace flagdnn::ascend {

class AscendError : public std::runtime_error {
 public:
  AscendError(flagdnnBackendResult_t result, std::string message);

  [[nodiscard]] flagdnnBackendResult_t result() const noexcept;

 private:
  flagdnnBackendResult_t result_;
};

[[nodiscard]] std::string acl_error(aclError result,
                                    const char* operation);
void check_acl(aclError result, const char* operation);
void require(bool condition,
             const char* message,
             flagdnnBackendResult_t result =
                 FLAGDNN_BACKEND_RESULT_INVALID_VALUE);

}  // namespace flagdnn::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_ERROR_HPP_
