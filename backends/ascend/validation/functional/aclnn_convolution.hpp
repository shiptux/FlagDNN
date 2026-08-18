/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_CONVOLUTION_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_CONVOLUTION_HPP_

#include "common/convolution.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing {

struct AclnnConvolutionFpropPlan {
  TestTensor input;
  TestTensor filter;
  TestTensor output;
  TestTensor convolution_input;
  std::vector<std::int64_t> explicit_padding;
  std::vector<std::int64_t> convolution_padding;
  std::vector<std::int64_t> stride;
  std::vector<std::int64_t> dilation;
  std::vector<std::int64_t> output_padding;
  std::int64_t groups = 1;
  std::int8_t cube_math_type = 0;
  bool transposed = false;

  [[nodiscard]] bool requires_explicit_padding() const noexcept {
    return !explicit_padding.empty();
  }
};

class AclnnConvolutionUnsupportedError final : public std::runtime_error {
 public:
  AclnnConvolutionUnsupportedError(std::int32_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

[[nodiscard]] bool aclnn_convolution_status_is_unsupported(
    std::int32_t status, std::string_view message);

[[nodiscard]] AclnnConvolutionFpropPlan plan_aclnn_convolution_fprop(
    const ConvolutionTestCase& test_case);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_CONVOLUTION_HPP_
