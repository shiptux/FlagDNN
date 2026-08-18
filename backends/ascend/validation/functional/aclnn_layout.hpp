/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_LAYOUT_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_LAYOUT_HPP_

#include "common/layout.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::testing {

struct AclnnLayoutPlan {
  LayoutOperation operation = LayoutOperation::kReshape;
  TestTensor input;
  TestTensor output;
  std::vector<std::int64_t> parameters;
  std::vector<std::int64_t> slice_begin;
  std::vector<std::int64_t> slice_end;
  std::vector<std::int64_t> collapsed_output_dimensions;
  std::vector<std::int64_t> collapsed_output_strides;
  std::int64_t flatten_axis = 0;
  bool materializes_reshape = false;
};

class AclnnLayoutUnsupportedError final : public std::runtime_error {
 public:
  AclnnLayoutUnsupportedError(std::int32_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

[[nodiscard]] AclnnLayoutPlan plan_aclnn_layout(
    const LayoutTestCase& test_case);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_FUNCTIONAL_ACLNN_LAYOUT_HPP_
