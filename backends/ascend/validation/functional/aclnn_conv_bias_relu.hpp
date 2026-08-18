/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_CONV_BIAS_RELU_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_CONV_BIAS_RELU_HPP_

#include "common/composite.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace flagdnn::testing {

struct AclnnConvBiasReluPlan {
  TestTensor x;
  TestTensor w;
  TestTensor bias;
  TestTensor convolution;
  TestTensor biased;
  TestTensor output;
};

class AclnnConvBiasReluUnsupportedError final : public std::runtime_error {
 public:
  AclnnConvBiasReluUnsupportedError(std::int32_t status,
                                    std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

[[nodiscard]] AclnnConvBiasReluPlan plan_aclnn_conv_bias_relu(
    const ConvBiasReluTestCase& test_case);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_CONV_BIAS_RELU_HPP_
