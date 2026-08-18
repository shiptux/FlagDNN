/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_REDUCTION_VALIDATION_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_REDUCTION_VALIDATION_HPP_

#include "common/reduction.hpp"

#include <span>
#include <vector>

namespace flagdnn::testing {

[[nodiscard]] std::vector<ReductionTestCase> make_ascend_reduction_cases(
    std::span<const ReductionTestCase> common_cases,
    flagdnnReductionMode_t mode);

[[nodiscard]] std::vector<float> make_reduction_input(
    const ReductionTestCase& test_case);

[[nodiscard]] std::vector<float> reduction_host_oracle(
    const ReductionTestCase& test_case,
    std::span<const float> logical_input);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_REDUCTION_VALIDATION_HPP_
