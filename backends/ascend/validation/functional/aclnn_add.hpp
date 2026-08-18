/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_ADD_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_ADD_HPP_

#include "common/add.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace flagdnn::testing {

struct AclnnAddPlan {
  TestTensor left;
  TestTensor right;
  TestTensor output;
  bool uses_contiguous_reference_output = false;
};

class AclnnUnsupportedError final : public std::runtime_error {
 public:
  AclnnUnsupportedError(std::int32_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

[[nodiscard]] AclnnAddPlan plan_aclnn_add(const AddTestCase& test_case);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_ADD_HPP_
