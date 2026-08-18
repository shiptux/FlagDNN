/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ENGINES_MATMUL_ORACLE_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ENGINES_MATMUL_ORACLE_HPP_

#include "backends/ascend/artifact.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace flagdnn::ascend {

void compute_matmul_host_oracle(const AscendStageArtifact& stage,
                                std::span<const std::uint8_t> a,
                                std::span<const std::uint8_t> b,
                                std::span<std::uint8_t> output);

[[nodiscard]] std::size_t matmul_kernel_input_count() noexcept;
[[nodiscard]] std::size_t matmul_output_argument_index() noexcept;
[[nodiscard]] std::size_t matmul_tensor_slot_count() noexcept;
[[nodiscard]] std::size_t matmul_runtime_argument_count() noexcept;
[[nodiscard]] const char* matmul_output_argument_name() noexcept;

void validate_matmul_stage_runtime_contract(const AscendStageArtifact& stage);
void seed_matmul_host_inputs(const AscendStageArtifact& stage,
                             std::span<std::uint8_t> a,
                             std::span<std::uint8_t> b);
void validate_matmul_host_output(const AscendStageArtifact& stage,
                                 std::span<const std::uint8_t> a,
                                 std::span<const std::uint8_t> b,
                                 std::span<const std::uint8_t> actual);
void validate_matmul_candidate_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> reference,
    std::span<const std::uint8_t> candidate);
void commit_matmul_host_output(const AscendStageArtifact& stage,
                               std::span<const std::uint8_t> actual,
                               std::span<std::uint8_t> shadow);

}  // namespace flagdnn::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_ENGINES_MATMUL_ORACLE_HPP_
