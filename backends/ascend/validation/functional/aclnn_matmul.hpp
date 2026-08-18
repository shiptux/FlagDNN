/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_MATMUL_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_MATMUL_HPP_

#include "common/matmul.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flagdnn::testing {

struct AclnnMatmulPlan {
  TestTensor a;
  TestTensor b;
  TestTensor output;
  std::int8_t cube_math_type = 0;
};

class AclnnMatmulUnsupportedError final : public std::runtime_error {
 public:
  AclnnMatmulUnsupportedError(std::int32_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

[[nodiscard]] bool aclnn_matmul_status_is_unsupported(
    std::int32_t status, std::string_view message);

[[nodiscard]] AclnnMatmulPlan plan_aclnn_matmul(
    const MatmulTestCase& test_case);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_MATMUL_HPP_
