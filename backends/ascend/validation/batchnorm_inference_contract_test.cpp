/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/cases.hpp"
#include "common/normalization.hpp"
#include "validation/batchnorm_inference_validation.hpp"
#include "validation/functional/aclnn_batchnorm_inference.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace validation = flagdnn::validation::ascend;
using flagdnn::testing::BatchnormInferenceTestCase;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template <typename Callback>
void require_failure(Callback&& callback, std::string_view needle) {
  try {
    callback();
  } catch (const std::exception& error) {
    require(std::string_view(error.what()).find(needle) !=
                std::string_view::npos,
            "BatchNorm inference failure did not identify its contract");
    return;
  }
  throw std::runtime_error("invalid BatchNorm inference case was accepted");
}

std::size_t storage_count(const flagdnn::testing::TestTensor& tensor) {
  std::size_t result = 1;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    result += static_cast<std::size_t>(tensor.dimensions[axis] - 1) *
              static_cast<std::size_t>(tensor.strides[axis]);
  }
  return result;
}

}  // namespace

int main() {
  try {
    require(flagdnn::testing::
                aclnn_batchnorm_inference_status_is_unsupported(
                    ACL_ERROR_API_NOT_SUPPORT,
                    "ACLNN BatchNorm inference is not supported"),
            "known ACLNN BatchNorm inference unsupported status was not "
            "classified");
    require(!flagdnn::testing::
                 aclnn_batchnorm_inference_status_is_unsupported(
                     123456,
                     "invalid BatchNorm inference descriptor"),
            "unexpected BatchNorm inference error was classified as "
            "unsupported");

    const std::vector<BatchnormInferenceTestCase> common =
        flagdnn::testing::make_batchnorm_inference_cases();
    require(common.size() == 12,
            "common BatchNorm inference catalog must contain 12 cases");
    const auto cases = validation::make_ascend_batchnorm_inference_cases(common);
    require(cases.size() == 18,
            "Ascend BatchNorm inference catalog must contain 18 cases");

    std::size_t local_rank2 = 0;
    std::size_t local_rank5 = 0;
    bool saw_gapped_offsets = false;
    for (std::size_t index = 0; index < cases.size(); ++index) {
      const BatchnormInferenceTestCase& test_case = cases[index];
      const validation::BatchnormInferencePlan plan =
          validation::plan_batchnorm_inference(test_case);
      require(plan.input.uid == test_case.x.uid &&
                  plan.weight.uid == test_case.scale.uid &&
                  plan.bias.uid == test_case.bias.uid &&
                  plan.mean.uid == test_case.mean.uid &&
                  plan.inverse_standard_deviation.uid ==
                      test_case.inv_variance.uid &&
                  plan.output.uid == test_case.y.uid,
              "ACLNN BatchNorm inference argument reorder is wrong");
      require(plan.input.dimensions == plan.output.dimensions &&
                  plan.input.data_type == plan.output.data_type,
              "BatchNorm inference X/Y contract changed during planning");
      require(plan.weight.data_type == FLAGDNN_DATA_FLOAT32 &&
                  plan.bias.data_type == FLAGDNN_DATA_FLOAT32 &&
                  plan.mean.data_type == FLAGDNN_DATA_FLOAT32 &&
                  plan.inverse_standard_deviation.data_type ==
                      FLAGDNN_DATA_FLOAT32,
              "BatchNorm inference parameters must stay FP32");
      if (index >= common.size()) {
        if (plan.input.dimensions.size() == 2) {
          ++local_rank2;
        } else if (plan.input.dimensions.size() == 5) {
          ++local_rank5;
          saw_gapped_offsets = saw_gapped_offsets ||
              (plan.input.binding_byte_offset == 64 &&
               plan.output.binding_byte_offset == 96 &&
               storage_count(plan.input) > 240 &&
               storage_count(plan.output) > 240);
        }
      }
    }
    require(local_rank2 == 3 && local_rank5 == 3,
            "Ascend local catalog must add rank-2/rank-5 for each dtype");
    require(saw_gapped_offsets,
            "Ascend local catalog lost gapped X/Y and aligned offsets");
    const auto rank2_reference =
        validation::batchnorm_inference_reference_data_tensor(
            cases.at(common.size()).x);
    require(rank2_reference.dimensions ==
                std::vector<std::int64_t>({4, 7, 1, 1}) &&
                rank2_reference.strides ==
                    std::vector<std::int64_t>({7, 1, 1, 1}) &&
                rank2_reference.binding_byte_offset == 0,
            "ACLNN rank-2 BatchNorm reference was not canonicalized to NCHW");
    const auto rank5_reference =
        validation::batchnorm_inference_reference_data_tensor(
            cases.at(common.size() + 3).x);
    require(rank5_reference.dimensions ==
                std::vector<std::int64_t>({2, 5, 1, 24}) &&
                rank5_reference.strides ==
                    std::vector<std::int64_t>({120, 24, 24, 1}) &&
                rank5_reference.binding_byte_offset == 0,
            "ACLNN rank-5 BatchNorm reference was not canonicalized to NCHW");

    const auto common_benchmarks =
        flagdnn::benchmarking::batchnorm_inference_benchmark_cases();
    require(common_benchmarks.size() == 24,
            "common BatchNorm inference benchmark must contain 24 cases");
    const auto benchmarks =
        validation::make_ascend_batchnorm_inference_benchmark_cases(
            common_benchmarks);
    require(benchmarks.size() == 25,
            "Ascend BatchNorm inference benchmark must contain 25 cases");
    const auto benchmark_plan =
        validation::plan_batchnorm_inference(benchmarks.back());
    require(benchmark_plan.input.data_type == FLAGDNN_DATA_FLOAT32 &&
                benchmark_plan.input.dimensions.size() == 5 &&
                storage_count(benchmark_plan.input) > 240 &&
                benchmark_plan.input.binding_byte_offset == 64 &&
                benchmark_plan.output.binding_byte_offset == 96,
            "Ascend benchmark extension must be FP32 rank-5 strided");
    const auto reference_benchmark =
        validation::batchnorm_inference_reference_benchmark_case(
            benchmarks.back());
    require(reference_benchmark.tensors[0].dimensions ==
                std::vector<std::int64_t>({2, 5, 1, 24}) &&
                reference_benchmark.tensors[0].strides ==
                    std::vector<std::int64_t>({120, 24, 24, 1}) &&
                reference_benchmark.tensors[0].binding_byte_offset == 0 &&
                reference_benchmark.tensors[5].dimensions ==
                    reference_benchmark.tensors[0].dimensions &&
                reference_benchmark.tensors[5].strides ==
                    reference_benchmark.tensors[0].strides &&
                reference_benchmark.tensors[5].binding_byte_offset == 0,
            "ACLNN benchmark reference case did not canonicalize only X/Y");
    for (std::size_t index = 1; index < 5; ++index) {
      const auto& actual = reference_benchmark.tensors[index];
      const auto& expected = benchmarks.back().tensors[index];
      require(actual.uid == expected.uid &&
                  actual.data_type == expected.data_type &&
                  actual.dimensions == expected.dimensions &&
                  actual.strides == expected.strides &&
                  actual.binding_byte_offset == expected.binding_byte_offset,
              "ACLNN benchmark reference changed a parameter tensor");
    }

    BatchnormInferenceTestCase oracle_case = cases.at(common.size() + 3);
    const validation::BatchnormInferencePlan oracle_plan =
        validation::plan_batchnorm_inference(oracle_case);
    std::vector<float> input(storage_count(oracle_plan.input), -777.0F);
    std::vector<float> output(storage_count(oracle_plan.output), -999.0F);
    const std::vector<float> output_before = output;
    const std::vector<float> mean = {100000.0F, -50000.0F, 3.0F,
                                     0.25F, -11.0F};
    const std::vector<float> invstd = {0.00001F, 64.0F, 0.5F, 2048.0F, 3.0F};
    const std::vector<float> scale = {2.0F, -0.25F, 4.0F, 0.5F, -2.0F};
    const std::vector<float> bias = {7.0F, 9.0F, -1.0F, 3.0F, 5.0F};
    for (std::size_t n = 0; n < 2; ++n) {
      for (std::size_t c = 0; c < 5; ++c) {
        for (std::size_t d = 0; d < 3; ++d) {
          for (std::size_t h = 0; h < 2; ++h) {
            for (std::size_t w = 0; w < 4; ++w) {
              const std::size_t physical =
                  n * static_cast<std::size_t>(oracle_plan.input.strides[0]) +
                  c * static_cast<std::size_t>(oracle_plan.input.strides[1]) +
                  d * static_cast<std::size_t>(oracle_plan.input.strides[2]) +
                  h * static_cast<std::size_t>(oracle_plan.input.strides[3]) +
                  w * static_cast<std::size_t>(oracle_plan.input.strides[4]);
              input[physical] = mean[c] + static_cast<float>(n + d + h + w);
            }
          }
        }
      }
    }
    const std::vector<float> input_before = input;
    validation::batchnorm_inference_host_oracle(
        oracle_plan, input, mean, invstd, scale, bias, output);
    require(input == input_before,
            "BatchNorm inference host oracle modified its input");
    for (std::size_t n = 0; n < 2; ++n) {
      for (std::size_t c = 0; c < 5; ++c) {
        const std::size_t input_physical =
            n * static_cast<std::size_t>(oracle_plan.input.strides[0]) +
            c * static_cast<std::size_t>(oracle_plan.input.strides[1]);
        const std::size_t output_physical =
            n * static_cast<std::size_t>(oracle_plan.output.strides[0]) +
            c * static_cast<std::size_t>(oracle_plan.output.strides[1]);
        const float expected =
            (input[input_physical] - mean[c]) * invstd[c] * scale[c] + bias[c];
        require(std::abs(output[output_physical] - expected) < 1.0e-6F,
                "BatchNorm inference formula added epsilon or mapped C wrong");
      }
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
      if (output_before[index] == -999.0F && output[index] != -999.0F) {
        bool logical = false;
        for (std::size_t n = 0; n < 2 && !logical; ++n) {
          for (std::size_t c = 0; c < 5 && !logical; ++c) {
            for (std::size_t d = 0; d < 3 && !logical; ++d) {
              for (std::size_t h = 0; h < 2 && !logical; ++h) {
                for (std::size_t w = 0; w < 4; ++w) {
                  logical = index ==
                      n * static_cast<std::size_t>(oracle_plan.output.strides[0]) +
                      c * static_cast<std::size_t>(oracle_plan.output.strides[1]) +
                      d * static_cast<std::size_t>(oracle_plan.output.strides[2]) +
                      h * static_cast<std::size_t>(oracle_plan.output.strides[3]) +
                      w * static_cast<std::size_t>(oracle_plan.output.strides[4]);
                  if (logical) break;
                }
              }
            }
          }
        }
        require(logical, "BatchNorm inference host oracle modified output gaps");
      }
    }

    BatchnormInferenceTestCase invalid = cases.front();
    invalid.x.dimensions = {8};
    invalid.x.strides = {1};
    invalid.y.dimensions = invalid.x.dimensions;
    invalid.y.strides = invalid.x.strides;
    require_failure([&] { (void)validation::plan_batchnorm_inference(invalid); },
                    "rank");
    invalid = cases.front();
    invalid.scale.data_type = FLAGDNN_DATA_FLOAT16;
    require_failure([&] { (void)validation::plan_batchnorm_inference(invalid); },
                    "FP32");
    invalid = cases.front();
    invalid.x.strides = {1, 1, 1, 1};
    require_failure([&] { (void)validation::plan_batchnorm_inference(invalid); },
                    "overlap");
    invalid = cases.front();
    invalid.bias.dimensions = {1, invalid.x.dimensions[1] + 1, 1, 1};
    require_failure([&] { (void)validation::plan_batchnorm_inference(invalid); },
                    "channel");

    std::cout << "Ascend BatchNorm inference validation contract passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
