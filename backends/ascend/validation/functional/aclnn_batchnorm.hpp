/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_BATCHNORM_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_BATCHNORM_HPP_

#include "validation/batchnorm_validation.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flagdnn::testing {

class AclnnBatchnormUnsupportedError final : public std::runtime_error {
 public:
  AclnnBatchnormUnsupportedError(std::int32_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

[[nodiscard]] bool aclnn_batchnorm_status_is_unsupported(
    std::int32_t status, std::string_view message);

struct AclnnBatchnormPlan {
  validation::ascend::BatchnormPlan operation;
  bool training = true;
  std::size_t binding_count = 10;
  std::size_t output_count = 5;
};

[[nodiscard]] AclnnBatchnormPlan plan_aclnn_batchnorm(
    const BatchnormTestCase& test_case);
[[nodiscard]] std::unique_ptr<NormalizationExecutable>
build_batchnorm_reference(const BatchnormTestCase& test_case);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_BATCHNORM_HPP_
