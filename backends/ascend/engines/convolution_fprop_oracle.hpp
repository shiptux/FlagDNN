/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ENGINES_CONVOLUTION_FPROP_ORACLE_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ENGINES_CONVOLUTION_FPROP_ORACLE_HPP_

#include "backends/ascend/artifact.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace flagdnn::ascend {

void compute_convolution_fprop_host_oracle(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> input,
    std::span<const std::uint8_t> filter,
    std::span<std::uint8_t> output);

[[nodiscard]] std::size_t convolution_fprop_kernel_input_count() noexcept;
[[nodiscard]] std::size_t convolution_fprop_output_argument_index() noexcept;
[[nodiscard]] std::size_t convolution_fprop_tensor_slot_count() noexcept;
[[nodiscard]] std::size_t convolution_fprop_runtime_argument_count() noexcept;
[[nodiscard]] const char* convolution_fprop_output_argument_name() noexcept;

void validate_convolution_fprop_stage_runtime_contract(
    const AscendStageArtifact& stage);
void seed_convolution_fprop_host_inputs(
    const AscendStageArtifact& stage,
    std::span<std::uint8_t> input,
    std::span<std::uint8_t> filter);
void validate_convolution_fprop_host_output(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> input,
    std::span<const std::uint8_t> filter,
    std::span<const std::uint8_t> actual);
void validate_convolution_fprop_candidate_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> reference,
    std::span<const std::uint8_t> candidate);
void commit_convolution_fprop_host_output(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> actual,
    std::span<std::uint8_t> shadow);

}  // namespace flagdnn::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_ENGINES_CONVOLUTION_FPROP_ORACLE_HPP_
