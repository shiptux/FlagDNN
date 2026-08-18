/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/artifact.hpp"
#include "backends/ascend/error.hpp"
#include "runtime/sha256.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::ascend {

/* This no-device harness needs the artifact error contract but no AscendCL
 * implementation. Keeping these definitions local avoids loading the driver
 * merely to validate immutable JSON and source files. */
AscendError::AscendError(flagdnnBackendResult_t result, std::string message)
    : std::runtime_error(std::move(message)), result_(result) {}

flagdnnBackendResult_t AscendError::result() const noexcept {
  return result_;
}

void require(bool condition,
             const char* message,
             flagdnnBackendResult_t result) {
  if (!condition) {
    throw AscendError(result, message == nullptr ? "invalid value" : message);
  }
}

}  // namespace flagdnn::ascend

namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open request fixture: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 7) {
    std::cerr << "usage: artifact_contract_driver TARGET AI_CORE_COUNT REQUEST ARTIFACT "
                 "success|compilation_failed CASE\n";
    return 2;
  }

  try {
    const std::string target = argv[1];
    const unsigned long encoded_ai_core_count = std::stoul(argv[2]);
    if (encoded_ai_core_count == 0 ||
        encoded_ai_core_count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("AI core count is outside uint32_t");
    }
    const auto ai_core_count =
        static_cast<std::uint32_t>(encoded_ai_core_count);
    const std::string request = read_file(argv[3]);
    const std::filesystem::path artifact_directory = argv[4];
    const std::string expectation = argv[5];
    const std::string case_name = argv[6];
    const std::string request_sha256 = flagdnn::native::sha256(request);
    const std::string artifact_path = artifact_directory.string();
    const flagdnnBackendBuildInputV2 input = {
        sizeof(flagdnnBackendBuildInputV2),
        request.data(),
        request.size(),
        artifact_path.c_str(),
        request_sha256.c_str(),
    };

    try {
      const flagdnn::ascend::AscendArtifact artifact =
          flagdnn::ascend::parse_ascend_artifact(
              target, ai_core_count, input);
      if (expectation != "success") {
        std::cerr << case_name << ": mutated artifact was accepted\n";
        return 1;
      }
      const bool large_channels_last =
          case_name == "valid_large_channels_last_dense_grid_cap";
      const bool unary = case_name.starts_with("valid_unary_");
      const bool sigmoid_backward =
          case_name.starts_with("valid_sigmoid_backward_");
      const bool logical_not =
          case_name.starts_with("valid_logical_not_");
      const bool logical_binary =
          case_name.starts_with("valid_logical_and_") ||
          case_name.starts_with("valid_logical_or_");
      const bool mixed_logical =
          case_name == "valid_mixed_logical_bool_dag";
      const bool comparison =
          case_name.starts_with("valid_comparison_");
      const bool mixed_comparison_logical =
          case_name == "valid_mixed_comparison_logical_bool_dag";
      const bool binary_select =
          case_name.starts_with("valid_binary_select_");
      const bool mixed_binary_select =
          case_name == "valid_mixed_binary_select_virtual_dag";
      const bool mixed_sigmoid_backward =
          case_name == "valid_mixed_sigmoid_backward_add_dag";
      const bool mixed_pointwise =
          case_name == "valid_mixed_binary_unary_dag";
      const bool mixed_unary_math =
          case_name == "valid_mixed_sqrt_rsqrt_reciprocal_dag" ||
          case_name == "valid_mixed_exp_log_tanh_dag";
      const bool mixed_activation =
          case_name == "valid_mixed_sigmoid_elu_softplus_swish_dag";
      const bool mixed_gelu =
          case_name == "valid_mixed_gelu_gelu_approx_tanh_dag";
      const bool layout = case_name.starts_with("valid_layout_");
      const bool scalar_layout =
          case_name == "valid_layout_scalar_reshape";
      const bool mixed_layout =
          case_name == "valid_mixed_layout_pointwise_virtual_dag";
      const bool mixed_reduction =
          case_name == "valid_reduction_layout_pointwise_virtual_dag";
      const bool reduction =
          case_name.starts_with("valid_reduction_") && !mixed_reduction;
      const bool matmul = case_name.starts_with("valid_matmul_");
      const bool convolution_fprop =
          case_name.starts_with("valid_convolution_fprop_");
      const bool mixed_conv_bias_relu =
          case_name == "valid_conv_bias_relu_fp16_autotuned";
      const bool convolution_fprop_autotuned =
          case_name == "valid_convolution_fprop_fp16_grouped_autotuned";
      const bool convolution_fprop_grouped =
          case_name.starts_with("valid_convolution_fprop_fp16_grouped_");
      const bool convolution_fprop_rank1 =
          case_name == "valid_convolution_fprop_fp32_rank1_channels_last_fixed";
      const bool convolution_fprop_rank3 =
          case_name == "valid_convolution_fprop_bf16_rank3_channels_last_fixed";
      const bool matmul_autotuned =
          case_name == "valid_matmul_fp16_autotuned";
      const bool matmul_broadcast =
          case_name == "valid_matmul_bf16_broadcast_gapped_fixed";
      const bool batchnorm =
          case_name.starts_with("valid_batchnorm_inference_") ||
          case_name.starts_with("valid_batchnorm_parameter_shape_");
      const bool batchnorm_training =
          case_name.starts_with("valid_batchnorm_training_");
      const bool batchnorm_training_autotuned =
          case_name == "valid_batchnorm_training_fp16_autotuned";
      const bool batchnorm_virtual =
          case_name == "valid_batchnorm_inference_virtual_dag_fixed";
      const bool batchnorm_fast =
          case_name == "valid_batchnorm_inference_fp16_fast_fixed" ||
          case_name == "valid_batchnorm_inference_fp16_fast_autotuned" ||
          case_name.starts_with("valid_batchnorm_parameter_shape_") ||
          batchnorm_virtual;
      const bool batchnorm_autotuned =
          case_name == "valid_batchnorm_inference_fp16_fast_autotuned";
      const bool rmsnorm = case_name.starts_with("valid_rmsnorm_");
      const bool rmsnorm_autotuned =
          case_name == "valid_rmsnorm_fp16_suffix1_autotuned";
      const bool layernorm = case_name.starts_with("valid_layernorm_");
      const bool layernorm_autotuned =
          case_name == "valid_layernorm_bf16_rank8_suffix3_autotuned";
      const bool layernorm_virtual =
          case_name == "valid_layernorm_fp16_virtual_y_fixed";
      const std::vector<std::int64_t> expected_bindings =
          mixed_reduction ? std::vector<std::int64_t>{1, 4}
          : mixed_conv_bias_relu ? std::vector<std::int64_t>{1, 2, 4, 6}
          : reduction ? std::vector<std::int64_t>{1, 2}
          : convolution_fprop ? std::vector<std::int64_t>{1, 2, 3}
          : matmul ? std::vector<std::int64_t>{1, 2, 3}
          : batchnorm_training
              ? std::vector<std::int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
          : layernorm_virtual ? std::vector<std::int64_t>{1, 2, 3, 5, 6}
          : layernorm ? std::vector<std::int64_t>{1, 2, 3, 4, 5, 6}
          : rmsnorm ? std::vector<std::int64_t>{1, 2, 3, 4, 5}
          : batchnorm_virtual
              ? std::vector<std::int64_t>{1, 3, 4, 5, 6, 8}
          : batchnorm ? std::vector<std::int64_t>{1, 2, 3, 4, 5, 6}
          : mixed_layout ? std::vector<std::int64_t>{1, 6}
          : layout ? std::vector<std::int64_t>{1, 2}
          : large_channels_last
              ? std::vector<std::int64_t>{1, 2, 3}
              : sigmoid_backward ? std::vector<std::int64_t>{1, 2, 3}
              : logical_not ? std::vector<std::int64_t>{1, 2}
              : logical_binary ? std::vector<std::int64_t>{1, 2, 3}
              : comparison ? std::vector<std::int64_t>{1, 2, 3}
              : binary_select ? std::vector<std::int64_t>{1, 2, 3, 4}
              : mixed_binary_select
                  ? std::vector<std::int64_t>{1, 2, 4, 5, 7}
              : mixed_comparison_logical
                  ? std::vector<std::int64_t>{1, 2, 4}
              : mixed_logical ? std::vector<std::int64_t>{1, 2, 5, 6}
              : unary ? std::vector<std::int64_t>{1, 2}
              : mixed_activation ? std::vector<std::int64_t>{1, 5}
              : mixed_gelu ? std::vector<std::int64_t>{1, 3}
              : mixed_unary_math ? std::vector<std::int64_t>{1, 4}
              : mixed_pointwise ? std::vector<std::int64_t>{1, 2, 5, 6}
                                : std::vector<std::int64_t>{1, 2, 4, 5};
      const std::size_t expected_stage_count =
          mixed_reduction ? 3
          : mixed_conv_bias_relu ? 3
          : reduction ? 1
          : convolution_fprop ? 1
          : matmul ? 1
          : batchnorm_training ? 1
          : layernorm ? 1
          : rmsnorm ? 1
          : batchnorm_virtual ? 3
          : batchnorm ? 1
          : mixed_layout ? 5
          : layout ? 1
          : large_channels_last || unary || sigmoid_backward || logical_not ||
                  logical_binary || comparison || binary_select
              ? 1
              : mixed_binary_select
                  ? 3
              : mixed_comparison_logical
                  ? 2
              : mixed_logical
                  ? 3
              : mixed_activation
                  ? 4
                  : mixed_gelu
                      ? 2
                  : mixed_pointwise || mixed_unary_math ? 3 : 2;
      const std::size_t expected_workspace_size =
          mixed_reduction ? 512
          : mixed_conv_bias_relu ? 512
          : reduction ? 0
          : convolution_fprop ? 0
          : matmul ? 0
          : batchnorm_training ? 0
          : layernorm_virtual ? 512
          : layernorm ? 0
          : rmsnorm ? 0
          : batchnorm_virtual ? 512
          : batchnorm ? 0
          : mixed_layout ? 1024
          : layout ? 0
          : large_channels_last || unary || sigmoid_backward || logical_not ||
                  logical_binary || comparison || binary_select
              ? 0
              : mixed_binary_select
                  ? 512
              : mixed_comparison_logical
                  ? 256
              : mixed_logical
                  ? 512
              : mixed_activation
                  ? 768
                  : mixed_gelu
                      ? 256
                  : mixed_pointwise || mixed_unary_math ? 512 : 256;
      if (artifact.stages.size() != expected_stage_count ||
          artifact.workspace_size != expected_workspace_size ||
          artifact.binding_uids != expected_bindings) {
        std::cerr << case_name
                  << ": valid artifact parsed to an unexpected plan\n";
        return 1;
      }
      if (unary && (artifact.stages[0].input_count != 1 ||
                    artifact.stages[0].arguments.size() != 3)) {
        std::cerr << case_name << ": unary raw ABI was not preserved\n";
        return 1;
      }
      if (layout &&
          (artifact.stages[0].kernel_family !=
               flagdnn::ascend::KernelFamily::kLayout ||
           artifact.stages[0].input_count != 1 ||
           artifact.stages[0].arguments.size() != 3 ||
           artifact.stages[0].arguments[0].name != "input_ptr" ||
           artifact.stages[0].arguments[1].name != "output_ptr" ||
           artifact.stages[0].arguments[2].name != "n_elements" ||
           artifact.stages[0].entry_point != "layout_copy_kernel")) {
        std::cerr << case_name << ": layout raw ABI was not preserved\n";
        return 1;
      }
      if (reduction &&
          (artifact.stages[0].kernel_family !=
               flagdnn::ascend::KernelFamily::kReduction ||
           artifact.stages[0].input_count != 1 ||
           artifact.stages[0].arguments.size() != 3 ||
           artifact.stages[0].arguments[0].name != "input_ptr" ||
           artifact.stages[0].arguments[1].name != "output_ptr" ||
           artifact.stages[0].arguments[2].name != "n_elements" ||
           artifact.stages[0].reduction_size <= 0 ||
           artifact.stages[0].n_elements <= 0)) {
        std::cerr << case_name << ": reduction raw ABI was not preserved\n";
        return 1;
      }
      if (matmul &&
          (artifact.stages[0].kernel_family !=
               flagdnn::ascend::KernelFamily::kMatMul ||
           artifact.stages[0].operation != "matmul" ||
           artifact.stages[0].input_count != 2 ||
           artifact.stages[0].arguments.size() != 4 ||
           artifact.stages[0].arguments[0].name != "a_ptr" ||
           artifact.stages[0].arguments[1].name != "b_ptr" ||
           artifact.stages[0].arguments[2].name != "output_ptr" ||
           artifact.stages[0].arguments[3].name != "n_elements" ||
           artifact.stages[0].n_elements != (matmul_broadcast ? 90 : 782) ||
           artifact.stages[0].matmul_batch != (matmul_broadcast ? 6 : 2) ||
           artifact.stages[0].matmul_m != (matmul_broadcast ? 3 : 17) ||
           artifact.stages[0].matmul_n != (matmul_broadcast ? 5 : 23) ||
           artifact.stages[0].matmul_k != (matmul_broadcast ? 4 : 30) ||
           artifact.stages[0].entry_point != "matmul_strided_kernel" ||
           artifact.stages[0].candidates.size() !=
               (matmul_autotuned ? 2U : 1U) ||
           (!matmul_autotuned &&
            artifact.stages[0].candidates[0].block_size != 16) ||
           (matmul_autotuned &&
            !((artifact.stages[0].candidates[0].block_size == 16 &&
               artifact.stages[0].candidates[1].block_size == 32) ||
              (artifact.stages[0].candidates[0].block_size == 32 &&
               artifact.stages[0].candidates[1].block_size == 16))))) {
        std::cerr << case_name << ": matmul raw ABI was not preserved\n";
        return 1;
      }
      if (convolution_fprop &&
          (artifact.stages[0].kernel_family !=
               flagdnn::ascend::KernelFamily::kConvolutionFprop ||
           artifact.stages[0].operation != "convolution_fprop" ||
           artifact.stages[0].input_count != 2 ||
           artifact.stages[0].arguments.size() != 4 ||
           artifact.stages[0].arguments[0].name != "input_ptr" ||
           artifact.stages[0].arguments[1].name != "filter_ptr" ||
           artifact.stages[0].arguments[2].name != "output_ptr" ||
           artifact.stages[0].arguments[3].name != "n_elements" ||
           artifact.stages[0].entry_point !=
               "convolution_fprop_persistent_kernel")) {
        std::cerr << case_name << ": convolution raw ABI was not preserved\n";
        return 1;
      }
      if (convolution_fprop_grouped &&
          (artifact.stages[0].n_elements != 192 ||
           artifact.stages[0].convolution_spatial_rank != 2 ||
           artifact.stages[0].convolution_groups != 2 ||
           artifact.stages[0].convolution_input_channels != 4 ||
           artifact.stages[0].convolution_output_channels != 6 ||
           artifact.stages[0].convolution_channels_per_group != 2 ||
           artifact.stages[0].convolution_input_dimensions !=
               std::array<std::int64_t, 5>{1, 4, 1, 7, 9} ||
           artifact.stages[0].convolution_input_strides !=
               std::array<std::int64_t, 5>{500, 100, 0, 12, 1} ||
           artifact.stages[0].convolution_filter_dimensions !=
               std::array<std::int64_t, 5>{6, 2, 1, 3, 2} ||
           artifact.stages[0].convolution_filter_strides !=
               std::array<std::int64_t, 5>{100, 30, 0, 8, 1} ||
           artifact.stages[0].convolution_output_dimensions !=
               std::array<std::int64_t, 5>{1, 6, 1, 4, 8} ||
           artifact.stages[0].convolution_output_strides !=
               std::array<std::int64_t, 5>{400, 60, 0, 10, 1} ||
           artifact.stages[0].convolution_pre_padding !=
               std::array<std::int64_t, 3>{0, 1, 0} ||
           artifact.stages[0].convolution_post_padding !=
               std::array<std::int64_t, 3>{0, 2, 1} ||
           artifact.stages[0].convolution_stride !=
               std::array<std::int64_t, 3>{1, 2, 1} ||
           artifact.stages[0].convolution_dilation !=
               std::array<std::int64_t, 3>{1, 1, 2})) {
        std::cerr << case_name
                  << ": grouped convolution metadata was not preserved\n";
        return 1;
      }
      if (convolution_fprop_rank1 &&
          (artifact.stages[0].n_elements != 192 ||
           artifact.stages[0].convolution_spatial_rank != 1 ||
           artifact.stages[0].convolution_groups != 1 ||
           artifact.stages[0].convolution_input_dimensions !=
               std::array<std::int64_t, 5>{2, 4, 1, 1, 16} ||
           artifact.stages[0].convolution_input_strides !=
               std::array<std::int64_t, 5>{64, 16, 0, 0, 1} ||
           artifact.stages[0].convolution_filter_dimensions !=
               std::array<std::int64_t, 5>{6, 4, 1, 1, 3} ||
           artifact.stages[0].convolution_output_dimensions !=
               std::array<std::int64_t, 5>{2, 6, 1, 1, 16} ||
           artifact.stages[0].convolution_output_strides !=
               std::array<std::int64_t, 5>{96, 1, 0, 0, 6} ||
           artifact.stages[0].convolution_pre_padding !=
               std::array<std::int64_t, 3>{0, 0, 1})) {
        std::cerr << case_name
                  << ": rank-1 convolution metadata was not preserved\n";
        return 1;
      }
      if (convolution_fprop_rank3 &&
          (artifact.stages[0].n_elements != 840 ||
           artifact.stages[0].convolution_spatial_rank != 3 ||
           artifact.stages[0].convolution_groups != 1 ||
           artifact.stages[0].convolution_input_dimensions !=
               std::array<std::int64_t, 5>{1, 2, 5, 6, 7} ||
           artifact.stages[0].convolution_input_strides !=
               std::array<std::int64_t, 5>{420, 1, 84, 14, 2} ||
           artifact.stages[0].convolution_filter_dimensions !=
               std::array<std::int64_t, 5>{4, 2, 3, 3, 3} ||
           artifact.stages[0].convolution_output_dimensions !=
               std::array<std::int64_t, 5>{1, 4, 5, 6, 7} ||
           artifact.stages[0].convolution_output_strides !=
               std::array<std::int64_t, 5>{840, 1, 168, 28, 4} ||
           artifact.stages[0].convolution_pre_padding !=
               std::array<std::int64_t, 3>{1, 1, 1})) {
        std::cerr << case_name
                  << ": rank-3 convolution metadata was not preserved\n";
        return 1;
      }
      if (convolution_fprop &&
          (artifact.stages[0].candidates.size() !=
               (convolution_fprop_autotuned ? 2U : 1U) ||
           (!convolution_fprop_autotuned &&
            artifact.stages[0].candidates[0].block_size != 256) ||
           (convolution_fprop_autotuned &&
            !((artifact.stages[0].candidates[0].block_size == 256 &&
               artifact.stages[0].candidates[1].block_size == 128) ||
              (artifact.stages[0].candidates[0].block_size == 128 &&
               artifact.stages[0].candidates[1].block_size == 256))))) {
        std::cerr << case_name << ": convolution candidates were not preserved\n";
        return 1;
      }
      if (rmsnorm &&
          (artifact.stages[0].kernel_family !=
               flagdnn::ascend::KernelFamily::kRmsNorm ||
           artifact.stages[0].input_count != 3 ||
           artifact.stages[0].arguments.size() != 6 ||
           artifact.stages[0].arguments[0].name != "x_ptr" ||
           artifact.stages[0].arguments[1].name != "scale_ptr" ||
           artifact.stages[0].arguments[2].name != "bias_ptr" ||
           artifact.stages[0].arguments[3].name != "y_ptr" ||
           artifact.stages[0].arguments[4].name != "inv_variance_ptr" ||
           artifact.stages[0].arguments[5].name != "n_elements" ||
           artifact.stages[0].n_elements != 170 ||
           artifact.stages[0].rmsnorm_rows != 10 ||
           artifact.stages[0].rmsnorm_normalized_elements != 17 ||
           artifact.stages[0].rmsnorm_epsilon != 1.0e-5 ||
           artifact.stages[0].entry_point != "rmsnorm_persistent_kernel" ||
           artifact.stages[0].candidates.size() !=
               (rmsnorm_autotuned ? 2U : 1U) ||
           (!rmsnorm_autotuned &&
            artifact.stages[0].candidates[0].block_size != 256) ||
           (rmsnorm_autotuned &&
            !((artifact.stages[0].candidates[0].block_size == 256 &&
               artifact.stages[0].candidates[1].block_size == 128) ||
              (artifact.stages[0].candidates[0].block_size == 128 &&
               artifact.stages[0].candidates[1].block_size == 256))))) {
        std::cerr << case_name << ": rmsnorm raw ABI was not preserved\n";
        return 1;
      }
      if (layernorm &&
          (artifact.stages[0].kernel_family !=
               flagdnn::ascend::KernelFamily::kLayerNorm ||
           artifact.stages[0].input_count != 3 ||
           artifact.stages[0].arguments.size() != 7 ||
           artifact.stages[0].arguments[0].name != "x_ptr" ||
           artifact.stages[0].arguments[1].name != "scale_ptr" ||
           artifact.stages[0].arguments[2].name != "bias_ptr" ||
           artifact.stages[0].arguments[3].name != "y_ptr" ||
           artifact.stages[0].arguments[4].name != "mean_ptr" ||
           artifact.stages[0].arguments[5].name != "inv_variance_ptr" ||
           artifact.stages[0].arguments[6].name != "n_elements" ||
           artifact.stages[0].n_elements !=
               (case_name == "valid_layernorm_fp32_suffix2_fixed"
                    ? 60
                    : case_name ==
                              "valid_layernorm_bf16_rank8_suffix3_autotuned"
                          ? 24
                          : 170) ||
           artifact.stages[0].layernorm_rows !=
               (case_name == "valid_layernorm_fp32_suffix2_fixed"
                    ? 3
                    : case_name ==
                              "valid_layernorm_bf16_rank8_suffix3_autotuned"
                          ? 4
                          : 10) ||
           artifact.stages[0].layernorm_normalized_elements !=
               (case_name == "valid_layernorm_fp32_suffix2_fixed"
                    ? 20
                    : case_name ==
                              "valid_layernorm_bf16_rank8_suffix3_autotuned"
                          ? 6
                          : 17) ||
           artifact.stages[0].layernorm_epsilon != 1.0e-5 ||
           artifact.stages[0].entry_point != "layernorm_persistent_kernel" ||
           artifact.stages[0].candidates.size() !=
               (layernorm_autotuned ? 2U : 1U) ||
           (!layernorm_autotuned &&
            artifact.stages[0].candidates[0].block_size != 256) ||
           (layernorm_autotuned &&
            !((artifact.stages[0].candidates[0].block_size == 256 &&
               artifact.stages[0].candidates[1].block_size == 128) ||
              (artifact.stages[0].candidates[0].block_size == 128 &&
               artifact.stages[0].candidates[1].block_size == 256))))) {
        std::cerr << case_name << ": layernorm raw ABI was not preserved\n";
        return 1;
      }
      if (batchnorm_training &&
          (artifact.stages[0].kernel_family !=
               flagdnn::ascend::KernelFamily::kBatchNorm ||
           artifact.stages[0].input_count != 5 ||
           artifact.stages[0].arguments.size() != 11 ||
           artifact.stages[0].arguments[0].name != "x_ptr" ||
           artifact.stages[0].arguments[1].name != "scale_ptr" ||
           artifact.stages[0].arguments[2].name != "bias_ptr" ||
           artifact.stages[0].arguments[3].name !=
               "previous_running_mean_ptr" ||
           artifact.stages[0].arguments[4].name !=
               "previous_running_variance_ptr" ||
           artifact.stages[0].arguments[5].name != "y_ptr" ||
           artifact.stages[0].arguments[6].name != "mean_ptr" ||
           artifact.stages[0].arguments[7].name != "inv_variance_ptr" ||
           artifact.stages[0].arguments[8].name !=
               "next_running_mean_ptr" ||
           artifact.stages[0].arguments[9].name !=
               "next_running_variance_ptr" ||
           artifact.stages[0].arguments[10].name != "n_elements" ||
           artifact.stages[0].n_elements != 24 ||
           artifact.stages[0].batchnorm_rank != 4 ||
           artifact.stages[0].batchnorm_batch != 2 ||
           artifact.stages[0].batchnorm_channels != 3 ||
           artifact.stages[0].batchnorm_spatial != 4 ||
           artifact.stages[0].batchnorm_reduction_elements != 8 ||
           artifact.stages[0].batchnorm_epsilon != 1.0e-3 ||
           artifact.stages[0].batchnorm_momentum != 0.1 ||
           artifact.stages[0].entry_point !=
               "batchnorm_training_persistent_kernel" ||
           artifact.stages[0].candidates.size() !=
               (batchnorm_training_autotuned ? 2U : 1U) ||
           (!batchnorm_training_autotuned &&
            artifact.stages[0].candidates[0].block_size != 256) ||
           (batchnorm_training_autotuned &&
            !((artifact.stages[0].candidates[0].block_size == 256 &&
               artifact.stages[0].candidates[1].block_size == 128) ||
              (artifact.stages[0].candidates[0].block_size == 128 &&
               artifact.stages[0].candidates[1].block_size == 256))))) {
        std::cerr << case_name
                  << ": batchnorm training raw ABI was not preserved\n";
        return 1;
      }
      const auto& batchnorm_stage =
          artifact.stages[batchnorm_virtual ? 1 : 0];
      if (batchnorm &&
          (batchnorm_stage.kernel_family !=
               flagdnn::ascend::KernelFamily::kBatchNormInference ||
           batchnorm_stage.input_count != 5 ||
           batchnorm_stage.arguments.size() != 7 ||
           batchnorm_stage.arguments[0].name != "x_ptr" ||
           batchnorm_stage.arguments[1].name != "mean_ptr" ||
           batchnorm_stage.arguments[2].name != "inv_variance_ptr" ||
           batchnorm_stage.arguments[3].name != "scale_ptr" ||
           batchnorm_stage.arguments[4].name != "bias_ptr" ||
           batchnorm_stage.arguments[5].name != "y_ptr" ||
           batchnorm_stage.arguments[6].name != "n_elements" ||
           batchnorm_stage.batchnorm_channels != 3 ||
           batchnorm_stage.batchnorm_rank !=
               (case_name == "valid_batchnorm_inference_fp32_generic_fixed"
                    ? 5
                    : case_name ==
                              "valid_batchnorm_inference_bf16_generic_fixed"
                          ? 3
                          : 2) ||
           batchnorm_stage.batchnorm_spatial !=
               (case_name == "valid_batchnorm_inference_fp32_generic_fixed"
                    ? 8
                    : case_name ==
                              "valid_batchnorm_inference_bf16_generic_fixed"
                          ? 2
                          : 1) ||
           batchnorm_stage.n_elements !=
               (case_name == "valid_batchnorm_inference_fp32_generic_fixed"
                    ? 48
                    : case_name ==
                              "valid_batchnorm_inference_bf16_generic_fixed"
                          ? 12
                          : 6) ||
           batchnorm_stage.entry_point !=
               (batchnorm_fast
                    ? "batchnorm_inference_nchw_persistent_kernel"
                    : "batchnorm_inference_strided_persistent_kernel") ||
           batchnorm_stage.candidates.size() !=
               (batchnorm_autotuned ? 2U : 1U) ||
           (!batchnorm_autotuned &&
            batchnorm_stage.candidates[0].block_size != 256) ||
           (batchnorm_autotuned &&
            !((batchnorm_stage.candidates[0].block_size == 256 &&
               batchnorm_stage.candidates[1].block_size == 128) ||
              (batchnorm_stage.candidates[0].block_size == 128 &&
               batchnorm_stage.candidates[1].block_size == 256))))) {
        std::cerr << case_name << ": batchnorm raw ABI was not preserved\n";
        return 1;
      }
      if (batchnorm_virtual &&
          (artifact.stages[0].dependencies.size() != 0 ||
           batchnorm_stage.dependencies != std::vector<std::size_t>{0} ||
           artifact.stages[2].dependencies != std::vector<std::size_t>{1} ||
           batchnorm_stage.arguments[0].source !=
               flagdnn::ascend::ArgumentSourceKind::kGraphWorkspace ||
           batchnorm_stage.arguments[0].workspace_offset != 0 ||
           batchnorm_stage.arguments[5].source !=
               flagdnn::ascend::ArgumentSourceKind::kGraphWorkspace ||
           batchnorm_stage.arguments[5].workspace_offset != 256 ||
           artifact.stages[2].arguments[0].source !=
               flagdnn::ascend::ArgumentSourceKind::kGraphWorkspace ||
           artifact.stages[2].arguments[0].workspace_offset != 256)) {
        std::cerr << case_name
                  << ": batchnorm virtual DAG contract was not preserved\n";
        return 1;
      }
      if (case_name == "valid_reduction_rank0_output" &&
          (artifact.stages[0].reduction_output_rank != 0 ||
           artifact.stages[0].n_elements != 1)) {
        std::cerr << case_name << ": scalar reduction was not preserved\n";
        return 1;
      }
      if (mixed_reduction &&
          (artifact.stages[0].kernel_family !=
               flagdnn::ascend::KernelFamily::kReduction ||
           artifact.stages[1].kernel_family !=
               flagdnn::ascend::KernelFamily::kLayout ||
           artifact.stages[2].kernel_family !=
               flagdnn::ascend::KernelFamily::kUnary ||
           artifact.stages[1].dependencies != std::vector<std::size_t>{0} ||
           artifact.stages[2].dependencies != std::vector<std::size_t>{1})) {
        std::cerr << case_name << ": mixed reduction DAG was not preserved\n";
        return 1;
      }
      if (mixed_conv_bias_relu &&
          (artifact.stages[0].kernel_family !=
               flagdnn::ascend::KernelFamily::kConvolutionFprop ||
           artifact.stages[1].kernel_family !=
               flagdnn::ascend::KernelFamily::kBinary ||
           artifact.stages[2].kernel_family !=
               flagdnn::ascend::KernelFamily::kUnary ||
           artifact.stages[0].dependencies != std::vector<std::size_t>{} ||
           artifact.stages[1].dependencies != std::vector<std::size_t>{0} ||
           artifact.stages[2].dependencies != std::vector<std::size_t>{1} ||
           artifact.stages[0].arguments[2].source !=
               flagdnn::ascend::ArgumentSourceKind::kGraphWorkspace ||
           artifact.stages[0].arguments[2].workspace_offset != 0 ||
           artifact.stages[1].arguments[0].source !=
               flagdnn::ascend::ArgumentSourceKind::kGraphWorkspace ||
           artifact.stages[1].arguments[0].workspace_offset != 0 ||
           artifact.stages[1].arguments[1].source !=
               flagdnn::ascend::ArgumentSourceKind::kBinding ||
           artifact.stages[1].arguments[1].uid != 4 ||
           artifact.stages[1].arguments[2].source !=
               flagdnn::ascend::ArgumentSourceKind::kGraphWorkspace ||
           artifact.stages[1].arguments[2].workspace_offset != 256 ||
           artifact.stages[2].arguments[0].source !=
               flagdnn::ascend::ArgumentSourceKind::kGraphWorkspace ||
           artifact.stages[2].arguments[0].workspace_offset != 256 ||
           artifact.stages[2].arguments[1].source !=
               flagdnn::ascend::ArgumentSourceKind::kBinding ||
           artifact.stages[2].arguments[1].uid != 6 ||
           artifact.stages[0].candidates.size() != 2 ||
           artifact.stages[1].candidates.size() != 2 ||
           artifact.stages[2].candidates.size() != 2)) {
        std::cerr << case_name
                  << ": ConvBiasRelu virtual DAG contract was not preserved\n";
        return 1;
      }
      if (scalar_layout &&
          (artifact.stages[0].layout_input_dimensions !=
               std::array<std::int64_t, 8>{1, 1, 1, 1, 1, 1, 1, 1} ||
           artifact.stages[0].layout_output_dimensions !=
               std::array<std::int64_t, 8>{1, 1, 1, 1, 1, 1, 1, 1})) {
        std::cerr << case_name << ": scalar layout padding was not preserved\n";
        return 1;
      }
      if (case_name == "valid_layout_slice_float32" &&
          (artifact.stages[0].layout_input_base != 9 ||
           artifact.stages[0].layout_input_strides[6] != 16 ||
           artifact.stages[0].layout_input_strides[7] != 2 ||
           artifact.stages[0].output_strides[6] != 4 ||
           artifact.stages[0].output_strides[7] != 1)) {
        std::cerr << case_name << ": slice mapping was not preserved\n";
        return 1;
      }
      if (mixed_layout &&
          (artifact.stages[0].kernel_family !=
               flagdnn::ascend::KernelFamily::kLayout ||
           artifact.stages[1].kernel_family !=
               flagdnn::ascend::KernelFamily::kUnary ||
           artifact.stages[2].kernel_family !=
               flagdnn::ascend::KernelFamily::kLayout ||
           artifact.stages[3].kernel_family !=
               flagdnn::ascend::KernelFamily::kLayout ||
           artifact.stages[4].kernel_family !=
               flagdnn::ascend::KernelFamily::kUnary)) {
        std::cerr << case_name << ": mixed layout DAG was not preserved\n";
        return 1;
      }
      if (sigmoid_backward &&
          (artifact.stages[0].input_count != 2 ||
           artifact.stages[0].arguments.size() != 4)) {
        std::cerr << case_name
                  << ": sigmoid_backward raw ABI was not preserved\n";
        return 1;
      }
      if (logical_not &&
          (artifact.stages[0].input_count != 1 ||
           artifact.stages[0].arguments.size() != 3 ||
           artifact.stages[0].tensor_storage_data_types !=
               std::vector<flagdnn::ascend::StorageDataType>(
                   2, flagdnn::ascend::StorageDataType::kBoolean))) {
        std::cerr << case_name
                  << ": logical_not BOOLEAN raw ABI was not preserved\n";
        return 1;
      }
      if (logical_binary &&
          (artifact.stages[0].input_count != 2 ||
           artifact.stages[0].arguments.size() != 4 ||
           artifact.stages[0].tensor_storage_data_types !=
               std::vector<flagdnn::ascend::StorageDataType>(
                   3, flagdnn::ascend::StorageDataType::kBoolean))) {
        std::cerr << case_name
                  << ": logical binary BOOLEAN raw ABI was not preserved\n";
        return 1;
      }
      if (mixed_logical &&
          (artifact.stages.size() != 3 ||
           artifact.stages[0].input_count != 2 ||
           artifact.stages[1].input_count != 1 ||
           artifact.stages[2].input_count != 2 ||
           artifact.stages[0].tensor_storage_data_types !=
               std::vector<flagdnn::ascend::StorageDataType>(
                   3, flagdnn::ascend::StorageDataType::kBoolean) ||
           artifact.stages[1].tensor_storage_data_types !=
               std::vector<flagdnn::ascend::StorageDataType>(
                   2, flagdnn::ascend::StorageDataType::kBoolean) ||
           artifact.stages[2].tensor_storage_data_types !=
               std::vector<flagdnn::ascend::StorageDataType>(
                   3, flagdnn::ascend::StorageDataType::kBoolean))) {
        std::cerr << case_name
                  << ": mixed logical BOOLEAN DAG was not preserved\n";
        return 1;
      }
      if (comparison &&
          (artifact.stages[0].input_count != 2 ||
           artifact.stages[0].arguments.size() != 4 ||
           artifact.stages[0].tensor_storage_data_types.size() != 3 ||
           artifact.stages[0].input_type(0) ==
               flagdnn::ascend::StorageDataType::kBoolean ||
           artifact.stages[0].input_type(0) !=
               artifact.stages[0].input_type(1) ||
           artifact.stages[0].output_type() !=
               flagdnn::ascend::StorageDataType::kBoolean)) {
        std::cerr << case_name
                  << ": comparison mixed storage ABI was not preserved\n";
        return 1;
      }
      if (binary_select &&
          (artifact.stages[0].input_count != 3 ||
           artifact.stages[0].arguments.size() != 5 ||
           artifact.stages[0].tensor_storage_data_types.size() != 4 ||
           artifact.stages[0].input_type(0) ==
               flagdnn::ascend::StorageDataType::kBoolean ||
           artifact.stages[0].input_type(0) !=
               artifact.stages[0].input_type(1) ||
           artifact.stages[0].input_type(2) !=
               flagdnn::ascend::StorageDataType::kBoolean ||
           artifact.stages[0].output_type() !=
               artifact.stages[0].input_type(0) ||
           artifact.stages[0].mask_strides[5] != 4 ||
           artifact.stages[0].mask_strides[6] != 1 ||
           artifact.stages[0].mask_strides[7] != 0)) {
        std::cerr << case_name
                  << ": binary_select mixed storage ABI was not preserved\n";
        return 1;
      }
      if (mixed_binary_select &&
          (artifact.stages.size() != 3 ||
           artifact.stages[0].input_count != 2 ||
           artifact.stages[0].output_type() !=
               flagdnn::ascend::StorageDataType::kBoolean ||
           artifact.stages[1].input_count != 3 ||
           artifact.stages[1].input_type(0) !=
               flagdnn::ascend::StorageDataType::kFloat32 ||
           artifact.stages[1].input_type(1) !=
               flagdnn::ascend::StorageDataType::kFloat32 ||
           artifact.stages[1].input_type(2) !=
               flagdnn::ascend::StorageDataType::kBoolean ||
           artifact.stages[1].output_type() !=
               flagdnn::ascend::StorageDataType::kFloat32 ||
           artifact.stages[2].input_count != 1)) {
        std::cerr << case_name
                  << ": binary_select virtual DAG was not preserved\n";
        return 1;
      }
      if (mixed_comparison_logical &&
          (artifact.stages.size() != 2 ||
           artifact.stages[0].input_type(0) !=
               flagdnn::ascend::StorageDataType::kFloat32 ||
           artifact.stages[0].input_type(1) !=
               flagdnn::ascend::StorageDataType::kFloat32 ||
           artifact.stages[0].output_type() !=
               flagdnn::ascend::StorageDataType::kBoolean ||
           artifact.stages[1].input_type(0) !=
               flagdnn::ascend::StorageDataType::kBoolean ||
           artifact.stages[1].output_type() !=
               flagdnn::ascend::StorageDataType::kBoolean)) {
        std::cerr << case_name
                  << ": comparison-to-logical virtual DAG was not preserved\n";
        return 1;
      }
      if (mixed_sigmoid_backward &&
          (artifact.stages[0].input_count != 2 ||
           artifact.stages[1].input_count != 2)) {
        std::cerr << case_name
                  << ": mixed sigmoid_backward arity was not preserved\n";
        return 1;
      }
      if (mixed_pointwise &&
          (artifact.stages[0].input_count != 2 ||
           artifact.stages[1].input_count != 1 ||
           artifact.stages[2].input_count != 2)) {
        std::cerr << case_name << ": mixed stage arity was not preserved\n";
        return 1;
      }
      if ((mixed_unary_math || mixed_activation || mixed_gelu) &&
          (artifact.stages[0].input_count != 1 ||
           artifact.stages[1].input_count != 1 ||
           (!mixed_gelu && artifact.stages[2].input_count != 1) ||
           (mixed_activation && artifact.stages[3].input_count != 1))) {
        std::cerr << case_name << ": unary math stage arity was not preserved\n";
        return 1;
      }
      return 0;
    } catch (const flagdnn::ascend::AscendError& error) {
      if (expectation == "compilation_failed" &&
          error.result() == FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED) {
        std::cout << case_name << ": " << error.what() << '\n';
        return 0;
      }
      std::cerr << case_name << ": unexpected result " << error.result()
                << ": " << error.what() << '\n';
      return 1;
    }
  } catch (const std::exception& error) {
    std::cerr << "artifact contract harness failed: " << error.what() << '\n';
    return 1;
  }
}
