/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ENGINES_BATCHNORM_INFERENCE_ORACLE_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ENGINES_BATCHNORM_INFERENCE_ORACLE_HPP_

#include "backends/ascend/artifact.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace flagdnn::ascend {

void compute_batchnorm_inference_host_oracle(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> x,
    std::span<const std::uint8_t> mean,
    std::span<const std::uint8_t> inv_variance,
    std::span<const std::uint8_t> scale,
    std::span<const std::uint8_t> bias,
    std::span<std::uint8_t> y);

[[nodiscard]] std::size_t batchnorm_inference_kernel_input_count() noexcept;
[[nodiscard]] std::size_t batchnorm_inference_tensor_slot_count() noexcept;
[[nodiscard]] std::size_t
batchnorm_inference_output_argument_index() noexcept;
[[nodiscard]] std::size_t
batchnorm_inference_runtime_argument_count() noexcept;
[[nodiscard]] const char* batchnorm_inference_output_argument_name() noexcept;

void validate_batchnorm_inference_stage_runtime_contract(
    const AscendStageArtifact& stage);
void seed_batchnorm_inference_host_inputs(
    const AscendStageArtifact& stage,
    std::span<std::uint8_t> x,
    std::span<std::uint8_t> mean,
    std::span<std::uint8_t> inv_variance,
    std::span<std::uint8_t> scale,
    std::span<std::uint8_t> bias);
void validate_batchnorm_inference_host_output(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> x,
    std::span<const std::uint8_t> mean,
    std::span<const std::uint8_t> inv_variance,
    std::span<const std::uint8_t> scale,
    std::span<const std::uint8_t> bias,
    std::span<const std::uint8_t> actual);
void validate_batchnorm_inference_candidate_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> reference,
    std::span<const std::uint8_t> candidate);
void commit_batchnorm_inference_host_output(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> actual,
    std::span<std::uint8_t> shadow);

}  // namespace flagdnn::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_ENGINES_BATCHNORM_INFERENCE_ORACLE_HPP_
