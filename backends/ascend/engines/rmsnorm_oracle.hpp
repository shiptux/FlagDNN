/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ENGINES_RMSNORM_ORACLE_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ENGINES_RMSNORM_ORACLE_HPP_

#include "backends/ascend/artifact.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace flagdnn::ascend {

[[nodiscard]] std::size_t rmsnorm_kernel_input_count() noexcept;
[[nodiscard]] std::size_t rmsnorm_tensor_slot_count() noexcept;
[[nodiscard]] std::size_t rmsnorm_runtime_argument_count() noexcept;
[[nodiscard]] std::size_t rmsnorm_y_argument_index() noexcept;
[[nodiscard]] std::size_t rmsnorm_inv_variance_argument_index() noexcept;

void validate_rmsnorm_stage_runtime_contract(const AscendStageArtifact& stage);
void seed_rmsnorm_host_inputs(const AscendStageArtifact& stage,
                              std::span<std::uint8_t> x,
                              std::span<std::uint8_t> scale,
                              std::span<std::uint8_t> bias);
void compute_rmsnorm_host_oracle(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> x,
    std::span<const std::uint8_t> scale,
    std::span<const std::uint8_t> bias,
    std::span<std::uint8_t> y,
    std::span<std::uint8_t> inv_variance);
void validate_rmsnorm_host_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> x,
    std::span<const std::uint8_t> scale,
    std::span<const std::uint8_t> bias,
    std::span<const std::uint8_t> y,
    std::span<const std::uint8_t> inv_variance);
void validate_rmsnorm_candidate_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> reference_y,
    std::span<const std::uint8_t> reference_inv_variance,
    std::span<const std::uint8_t> candidate_y,
    std::span<const std::uint8_t> candidate_inv_variance);
void commit_rmsnorm_host_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> actual_y,
    std::span<const std::uint8_t> actual_inv_variance,
    std::span<std::uint8_t> shadow_y,
    std::span<std::uint8_t> shadow_inv_variance);

}  // namespace flagdnn::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_ENGINES_RMSNORM_ORACLE_HPP_
