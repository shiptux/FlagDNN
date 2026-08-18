/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ENGINES_BATCHNORM_ORACLE_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ENGINES_BATCHNORM_ORACLE_HPP_

#include "backends/ascend/artifact.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace flagdnn::ascend {

using BatchNormHostBuffers =
    std::array<std::span<std::uint8_t>, 5>;
using ConstBatchNormHostBuffers =
    std::array<std::span<const std::uint8_t>, 5>;

[[nodiscard]] std::size_t batchnorm_kernel_input_count() noexcept;
[[nodiscard]] std::size_t batchnorm_tensor_slot_count() noexcept;
[[nodiscard]] std::size_t batchnorm_runtime_argument_count() noexcept;
[[nodiscard]] std::size_t batchnorm_first_output_argument_index() noexcept;

void validate_batchnorm_stage_runtime_contract(const AscendStageArtifact& stage);
void seed_batchnorm_host_inputs(const AscendStageArtifact& stage,
                                BatchNormHostBuffers inputs);
void compute_batchnorm_host_oracle(const AscendStageArtifact& stage,
                                   ConstBatchNormHostBuffers inputs,
                                   BatchNormHostBuffers outputs);
void validate_batchnorm_host_outputs(const AscendStageArtifact& stage,
                                     ConstBatchNormHostBuffers inputs,
                                     ConstBatchNormHostBuffers outputs);
void validate_batchnorm_candidate_outputs(
    const AscendStageArtifact& stage,
    ConstBatchNormHostBuffers reference,
    ConstBatchNormHostBuffers candidate);
void commit_batchnorm_host_outputs(const AscendStageArtifact& stage,
                                   ConstBatchNormHostBuffers actual,
                                   BatchNormHostBuffers shadow);

}  // namespace flagdnn::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_ENGINES_BATCHNORM_ORACLE_HPP_
