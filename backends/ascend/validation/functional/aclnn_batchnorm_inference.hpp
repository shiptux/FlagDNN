/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_BATCHNORM_INFERENCE_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_BATCHNORM_INFERENCE_HPP_

#include "common/normalization.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flagdnn::testing {

class AclnnBatchnormInferenceUnsupportedError final
    : public std::runtime_error {
 public:
  AclnnBatchnormInferenceUnsupportedError(std::int32_t status,
                                          std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

[[nodiscard]] bool aclnn_batchnorm_inference_status_is_unsupported(
    std::int32_t status, std::string_view message);

[[nodiscard]] std::unique_ptr<NormalizationExecutable>
build_batchnorm_inference_reference(
    const BatchnormInferenceTestCase& test_case);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_BATCHNORM_INFERENCE_HPP_
