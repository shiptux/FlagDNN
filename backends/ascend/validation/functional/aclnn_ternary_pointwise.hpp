/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_TERNARY_POINTWISE_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_TERNARY_POINTWISE_HPP_

#include "common/pointwise.hpp"

#include <memory>

namespace flagdnn::testing {

struct AclnnTernaryPointwisePlan {
  TestTensor self;
  TestTensor other;
  TestTensor condition;
  TestTensor output;
};

[[nodiscard]] AclnnTernaryPointwisePlan plan_aclnn_ternary_pointwise(
    const PointwiseTestCase& test_case);

[[nodiscard]] std::unique_ptr<PointwiseExecutable>
build_aclnn_ternary_pointwise_reference(
    const PointwiseTestCase& test_case);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_TERNARY_POINTWISE_HPP_
