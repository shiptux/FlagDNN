/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_UNARY_POINTWISE_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_UNARY_POINTWISE_HPP_

#include "common/pointwise.hpp"

#include <memory>

namespace flagdnn::testing {

struct AclnnUnaryPointwisePlan {
  TestTensor input;
  TestTensor output;
  bool uses_contiguous_reference_output = false;
};

[[nodiscard]] AclnnUnaryPointwisePlan plan_aclnn_unary_pointwise(
    const PointwiseTestCase& test_case);

[[nodiscard]] std::unique_ptr<PointwiseExecutable>
build_aclnn_unary_pointwise_reference(const PointwiseTestCase& test_case);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_UNARY_POINTWISE_HPP_
