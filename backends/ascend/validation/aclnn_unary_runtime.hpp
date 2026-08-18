/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_ACLNN_UNARY_RUNTIME_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_ACLNN_UNARY_RUNTIME_HPP_

#include <flagdnn/flagdnn.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::validation::ascend {

struct AclnnUnaryTensor {
  flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT32;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
};

class AclnnUnaryUnsupportedError final : public std::runtime_error {
 public:
  AclnnUnaryUnsupportedError(std::int32_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_ = 0;
};

class AclnnUnaryRuntime final {
 public:
  AclnnUnaryRuntime(flagdnnPointwiseMode_t mode,
                    flagdnnPointwiseAttributes_t attributes);
  ~AclnnUnaryRuntime();

  AclnnUnaryRuntime(const AclnnUnaryRuntime&) = delete;
  AclnnUnaryRuntime& operator=(const AclnnUnaryRuntime&) = delete;

  void prepare(const AclnnUnaryTensor& input,
               void* input_pointer,
               const AclnnUnaryTensor& output,
               void* output_pointer,
               flagdnnStream_t stream);

  [[nodiscard]] std::size_t workspace_size() const noexcept;

  void execute(void* input_pointer,
               void* output_pointer,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream);

 private:
  struct State;

  flagdnnPointwiseMode_t mode_ = FLAGDNN_POINTWISE_NOT_SET;
  flagdnnPointwiseAttributes_t attributes_ =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  std::unique_ptr<State> state_;
};

[[nodiscard]] bool is_aclnn_unary_mode(flagdnnPointwiseMode_t mode) noexcept;

}  // namespace flagdnn::validation::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_ACLNN_UNARY_RUNTIME_HPP_
