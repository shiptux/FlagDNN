/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/reduction.hpp"
#include "common/cases.hpp"
#include "validation/benchmark/aclnn_provider.hpp"
#include "validation/functional/aclnn_reduction.hpp"
#include "validation/reduction_validation.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;
using flagdnn::testing::AclnnReductionPlan;
using flagdnn::testing::AclnnReductionUnsupportedError;
using flagdnn::testing::ReductionTestCase;

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
            "reduction failure did not identify the violated contract");
    return;
  }
  throw std::runtime_error("invalid reduction plan was accepted");
}

}  // namespace

int main() {
  try {
    require(flagdnn::testing::aclnn_reduction_status_is_unsupported(
                ACL_ERROR_API_NOT_SUPPORT,
                "ACLNN API is not supported"),
            "known ACLNN reduction unsupported status was not classified");
    require(!flagdnn::testing::aclnn_reduction_status_is_unsupported(
                123456,
                "invalid reduction descriptor"),
            "unexpected ACLNN reduction error was classified as unsupported");
    try {
      throw AclnnReductionUnsupportedError(ACL_ERROR_API_NOT_SUPPORT,
                                           "unsupported reduction");
    } catch (const AclnnReductionUnsupportedError& error) {
      require(error.status() == ACL_ERROR_API_NOT_SUPPORT,
              "typed ACLNN reduction unsupported status was not preserved");
    }

    const std::vector<ReductionTestCase> cases =
        flagdnn::testing::make_reduction_cases();
    require(cases.size() == 23,
            "common reduction catalog must contain 23 cases");

    std::size_t sum_count = 0;
    std::size_t average_count = 0;
    std::size_t product_count = 0;
    bool saw_negative_axis = false;
    bool saw_unaligned_input = false;
    bool saw_scalar_output = false;
    for (const ReductionTestCase& test_case : cases) {
      const AclnnReductionPlan plan =
          flagdnn::testing::plan_aclnn_reduction(test_case);
      require(plan.mode == test_case.mode,
              "ACLNN reduction plan changed the reduction mode");
      require(plan.input.uid == test_case.input.uid &&
                  plan.input.data_type == test_case.input.data_type &&
                  plan.input.dimensions == test_case.input.dimensions &&
                  plan.input.strides == test_case.input.strides,
              "ACLNN reduction plan changed the reference input layout");
      require(plan.input.binding_byte_offset == 0,
              "ACLNN reference input must use an aligned binding entrance");
      require(plan.output.uid == test_case.output.uid &&
                  plan.output.data_type == test_case.output.data_type &&
                  plan.output.dimensions == test_case.output.dimensions &&
                  plan.output.strides == test_case.output.strides &&
                  plan.output.binding_byte_offset ==
                      test_case.output.binding_byte_offset,
              "ACLNN reduction plan changed the output contract");
      require(plan.axis >= 0 &&
                  static_cast<std::size_t>(plan.axis) <
                      plan.input.dimensions.size(),
              "ACLNN reduction plan did not normalize the axis");
      require(plan.keep_dimensions == test_case.keep_dimensions,
              "ACLNN reduction plan changed keep-dimensions");

      saw_negative_axis = saw_negative_axis || test_case.axis < 0;
      saw_unaligned_input =
          saw_unaligned_input || test_case.input.binding_byte_offset != 0;
      if (test_case.output.dimensions.empty()) {
        saw_scalar_output = true;
        require(tensor_io::element_count(plan.output) == 1 &&
                    tensor_io::storage_element_count(plan.output) == 1,
                "ACLNN scalar reduction output is not one element");
      }
      switch (test_case.mode) {
        case FLAGDNN_REDUCTION_ADD:
          ++sum_count;
          break;
        case FLAGDNN_REDUCTION_AVG:
          ++average_count;
          break;
        case FLAGDNN_REDUCTION_MUL:
          ++product_count;
          break;
      }
    }
    require(sum_count == 11 && average_count == 6 && product_count == 6,
            "common reduction modes must split 11/6/6");
    require(saw_negative_axis,
            "common reduction catalog lost its negative-axis case");
    require(saw_unaligned_input,
            "common reduction catalog lost its unaligned-input cases");
    require(saw_scalar_output,
            "common reduction catalog lost its rank-0 output case");

    const auto benchmark_cases =
        flagdnn::benchmarking::reduction_benchmark_cases();
    require(benchmark_cases.size() == 9,
            "reduction benchmark catalog must contain 9 cases");
    std::size_t benchmark_sum = 0;
    std::size_t benchmark_average = 0;
    std::size_t benchmark_product = 0;
    bool saw_bfloat16_product = false;
    for (const auto& benchmark_case : benchmark_cases) {
      const auto benchmark_plan =
          flagdnn::benchmarking::plan_aclnn_reduction(
              benchmark_case);
      require(benchmark_plan.input.uid == benchmark_case.tensors[0].uid &&
                  benchmark_plan.input.dimensions ==
                      benchmark_case.tensors[0].dimensions &&
                  benchmark_plan.input.strides ==
                      benchmark_case.tensors[0].strides &&
                  benchmark_plan.input.binding_byte_offset == 0 &&
                  benchmark_plan.output.uid == benchmark_case.tensors[1].uid &&
                  benchmark_plan.output.dimensions ==
                      benchmark_case.tensors[1].dimensions &&
                  benchmark_plan.output.strides ==
                      benchmark_case.tensors[1].strides &&
                  benchmark_plan.axis == 1 &&
                  benchmark_plan.keep_dimensions,
              "ACLNN reduction benchmark plan changed the case contract");
      switch (benchmark_plan.mode) {
        case FLAGDNN_REDUCTION_ADD:
          ++benchmark_sum;
          break;
        case FLAGDNN_REDUCTION_AVG:
          ++benchmark_average;
          break;
        case FLAGDNN_REDUCTION_MUL:
          ++benchmark_product;
          saw_bfloat16_product =
              saw_bfloat16_product ||
              benchmark_plan.input.data_type == FLAGDNN_DATA_BFLOAT16;
          break;
      }
    }
    require(benchmark_sum == 3 && benchmark_average == 3 &&
                benchmark_product == 3,
            "reduction benchmark modes must split 3/3/3");
    require(saw_bfloat16_product,
            "BF16 product must reach the required ACLNN device gate");

    const auto sum_cases = flagdnn::testing::make_ascend_reduction_cases(
        cases, FLAGDNN_REDUCTION_ADD);
    const auto average_cases = flagdnn::testing::make_ascend_reduction_cases(
        cases, FLAGDNN_REDUCTION_AVG);
    const auto product_cases = flagdnn::testing::make_ascend_reduction_cases(
        cases, FLAGDNN_REDUCTION_MUL);
    require(sum_cases.size() == 12 && average_cases.size() == 7 &&
                product_cases.size() == 7,
            "Ascend reduction catalogs must contain 12/7/7 cases");
    for (const auto* catalog : {&sum_cases, &average_cases, &product_cases}) {
      const ReductionTestCase& special = catalog->back();
      require(special.input.data_type == FLAGDNN_DATA_FLOAT32 &&
                  special.input.binding_byte_offset == 32 &&
                  special.output.binding_byte_offset == 64 &&
                  tensor_io::storage_element_count(special.input) >
                      tensor_io::element_count(special.input) &&
                  tensor_io::storage_element_count(special.output) >
                      tensor_io::element_count(special.output),
              "Ascend reduction special case must cover FP32 gaps and offsets");
    }
    const std::vector<float> oracle_input = {1.0F, 2.0F, 3.0F,
                                             4.0F, 5.0F, 6.0F};
    ReductionTestCase oracle_case;
    oracle_case.name = "oracle";
    oracle_case.input = {900, FLAGDNN_DATA_FLOAT32, {2, 3}, {3, 1}, 0};
    oracle_case.output = {901, FLAGDNN_DATA_FLOAT32, {2}, {1}, 0};
    oracle_case.axis = -1;
    oracle_case.keep_dimensions = false;
    oracle_case.mode = FLAGDNN_REDUCTION_ADD;
    oracle_case.absolute_tolerance = 0.0;
    oracle_case.relative_tolerance = 0.0;
    require(flagdnn::testing::reduction_host_oracle(
                oracle_case, oracle_input) == std::vector<float>({6.0F, 15.0F}),
            "FP32 sum host oracle is wrong");
    oracle_case.mode = FLAGDNN_REDUCTION_AVG;
    require(flagdnn::testing::reduction_host_oracle(
                oracle_case, oracle_input) == std::vector<float>({2.0F, 5.0F}),
            "FP32 average host oracle is wrong");
    oracle_case.mode = FLAGDNN_REDUCTION_MUL;
    require(flagdnn::testing::reduction_host_oracle(
                oracle_case, oracle_input) == std::vector<float>({6.0F, 120.0F}),
            "FP32 product host oracle is wrong");

    ReductionTestCase scalar_oracle;
    scalar_oracle.name = "scalar_oracle";
    scalar_oracle.input = {902, FLAGDNN_DATA_FLOAT32, {3}, {1}, 0};
    scalar_oracle.output = {903, FLAGDNN_DATA_FLOAT32, {}, {}, 0};
    scalar_oracle.mode = FLAGDNN_REDUCTION_ADD;
    scalar_oracle.axis = 0;
    scalar_oracle.keep_dimensions = false;
    scalar_oracle.absolute_tolerance = 0.0;
    scalar_oracle.relative_tolerance = 0.0;
    require(flagdnn::testing::reduction_host_oracle(
                scalar_oracle, std::span<const float>(oracle_input).first(3)) ==
                std::vector<float>({6.0F}),
            "rank-0 sum host oracle is wrong");

    ReductionTestCase keepdim_oracle;
    keepdim_oracle.name = "keepdim_oracle";
    keepdim_oracle.input = {
        904, FLAGDNN_DATA_FLOAT32, {2, 2, 2}, {4, 2, 1}, 0};
    keepdim_oracle.output = {
        905, FLAGDNN_DATA_FLOAT32, {2, 1, 2}, {2, 2, 1}, 0};
    keepdim_oracle.mode = FLAGDNN_REDUCTION_ADD;
    keepdim_oracle.axis = 1;
    keepdim_oracle.keep_dimensions = true;
    keepdim_oracle.absolute_tolerance = 0.0;
    keepdim_oracle.relative_tolerance = 0.0;
    const std::vector<float> keepdim_input = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    require(flagdnn::testing::reduction_host_oracle(
                keepdim_oracle, keepdim_input) ==
                std::vector<float>({4.0F, 6.0F, 12.0F, 14.0F}),
            "keep-dimensions sum host oracle mapped coordinates incorrectly");

    ReductionTestCase invalid = cases.front();
    invalid.mode = static_cast<flagdnnReductionMode_t>(99);
    require_failure(
        [&] { (void)flagdnn::testing::plan_aclnn_reduction(invalid); },
        "mode");
    std::cout << "Ascend reduction validation contract: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Ascend reduction validation contract failed: "
              << error.what() << '\n';
    return 1;
  }
}
