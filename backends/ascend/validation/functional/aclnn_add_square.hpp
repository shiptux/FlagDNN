/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_ADD_SQUARE_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_ADD_SQUARE_HPP_

#include "common/composite.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace flagdnn::testing {

struct AclnnAddSquarePlan {
  TestTensor left;
  TestTensor right;
  TestTensor right_alias;
  TestTensor square;
  TestTensor output;
};

class AclnnAddSquareUnsupportedError final : public std::runtime_error {
 public:
  AclnnAddSquareUnsupportedError(std::int32_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

[[nodiscard]] AclnnAddSquarePlan plan_aclnn_add_square(
    const AddSquareTestCase& test_case);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_ADD_SQUARE_HPP_
