/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ENGINES_REDUCTION_ORACLE_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ENGINES_REDUCTION_ORACLE_HPP_

#include "backends/ascend/artifact.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace flagdnn::ascend {

void compute_reduction_host_oracle(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> input,
    std::span<std::uint8_t> output);

[[nodiscard]] std::size_t reduction_kernel_input_count() noexcept;
[[nodiscard]] std::size_t reduction_output_argument_index() noexcept;
[[nodiscard]] std::size_t reduction_runtime_argument_count() noexcept;
[[nodiscard]] const char* reduction_output_argument_name() noexcept;
void validate_reduction_stage_runtime_contract(const AscendStageArtifact& stage);
void seed_reduction_host_input(const AscendStageArtifact& stage,
                               std::span<std::uint8_t> input);
void validate_reduction_host_output(const AscendStageArtifact& stage,
                                    std::span<const std::uint8_t> input,
                                    std::span<const std::uint8_t> actual);
void commit_reduction_host_output(const AscendStageArtifact& stage,
                                  std::span<const std::uint8_t> actual,
                                  std::span<std::uint8_t> shadow);

}  // namespace flagdnn::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_ENGINES_REDUCTION_ORACLE_HPP_
