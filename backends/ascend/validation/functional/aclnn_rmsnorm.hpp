/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_RMSNORM_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_RMSNORM_HPP_

#include "common/normalization.hpp"
#include "validation/rmsnorm_validation.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flagdnn::testing {

class AclnnRmsnormUnsupportedError final : public std::runtime_error {
 public:
  AclnnRmsnormUnsupportedError(std::int32_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

[[nodiscard]] bool aclnn_rmsnorm_status_is_unsupported(
    std::int32_t status, std::string_view message);

struct AclnnRmsnormPlan {
  validation::ascend::RmsnormPlan operation;
  TestTensor gamma;
  TestTensor normalized_output;
  std::size_t normalized_output_bytes = 0;
};

[[nodiscard]] AclnnRmsnormPlan plan_aclnn_rmsnorm(
    const RmsnormTestCase& test_case);

[[nodiscard]] std::unique_ptr<NormalizationExecutable>
build_rmsnorm_reference(const RmsnormTestCase& test_case);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_RMSNORM_HPP_
