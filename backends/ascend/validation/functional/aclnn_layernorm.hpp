/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_LAYERNORM_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_LAYERNORM_HPP_

#include "common/normalization.hpp"
#include "validation/layernorm_validation.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing {

class AclnnLayernormUnsupportedError final : public std::runtime_error {
 public:
  AclnnLayernormUnsupportedError(std::int32_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

[[nodiscard]] bool aclnn_layernorm_status_is_unsupported(
    std::int32_t status, std::string_view message);

struct AclnnLayernormPlan {
  validation::ascend::LayernormPlan operation;
  std::vector<std::int64_t> normalized_shape;
};

[[nodiscard]] AclnnLayernormPlan plan_aclnn_layernorm(
    const LayernormTestCase& test_case);

[[nodiscard]] std::unique_ptr<NormalizationExecutable>
build_layernorm_reference(const LayernormTestCase& test_case);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_LAYERNORM_HPP_
