/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/cases.hpp"
#include "common/matmul.hpp"
#include "validation/benchmark/aclnn_matmul_provider.hpp"
#include "validation/functional/aclnn_matmul.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>

#include <array>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;
using flagdnn::testing::AclnnMatmulPlan;
using flagdnn::testing::AclnnMatmulUnsupportedError;
using flagdnn::testing::MatmulTestCase;
using flagdnn::testing::TestTensor;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

bool same_tensor(const TestTensor& left, const TestTensor& right) {
  return left.uid == right.uid && left.data_type == right.data_type &&
         left.dimensions == right.dimensions &&
         left.strides == right.strides &&
         left.binding_byte_offset == right.binding_byte_offset;
}

template <typename Callback>
void require_failure(Callback&& callback, std::string_view needle) {
  try {
    callback();
  } catch (const std::exception& error) {
    require(std::string_view(error.what()).find(needle) !=
                std::string_view::npos,
            "MatMul failure did not identify the violated contract");
    return;
  }
  throw std::runtime_error("invalid MatMul ACLNN plan was accepted");
}

}  // namespace

int main() {
  try {
    require(flagdnn::testing::aclnn_matmul_status_is_unsupported(
                ACL_ERROR_API_NOT_SUPPORT,
                "ACLNN MatMul is not supported"),
            "known ACLNN MatMul unsupported status was not classified");
    require(!flagdnn::testing::aclnn_matmul_status_is_unsupported(
                123456,
                "invalid MatMul descriptor"),
            "unexpected ACLNN MatMul error was classified as unsupported");
    try {
      throw AclnnMatmulUnsupportedError(ACL_ERROR_API_NOT_SUPPORT,
                                        "unsupported MatMul");
    } catch (const AclnnMatmulUnsupportedError& error) {
      require(error.status() == ACL_ERROR_API_NOT_SUPPORT,
              "typed ACLNN MatMul unsupported status was not preserved");
    }

    const std::vector<MatmulTestCase> cases =
        flagdnn::testing::make_matmul_cases();
    require(cases.size() == 27,
            "common MatMul functional catalog must contain 27 cases");

    std::array<std::size_t, 3> type_counts{};
    std::size_t strided_count = 0;
    std::size_t autotune_count = 0;
    bool saw_matrix = false;
    bool saw_batched = false;
    bool saw_multidimensional_broadcast = false;
    for (const MatmulTestCase& test_case : cases) {
      const AclnnMatmulPlan plan =
          flagdnn::testing::plan_aclnn_matmul(test_case);
      require(same_tensor(plan.a, test_case.a) &&
                  same_tensor(plan.b, test_case.b) &&
                  same_tensor(plan.output, test_case.output),
              "ACLNN MatMul plan changed a tensor contract");
      require(plan.cube_math_type == 0,
              "ACLNN MatMul must use KEEP_DTYPE cube math");
      require(plan.a.binding_byte_offset == 0 &&
                  plan.b.binding_byte_offset == 0 &&
                  plan.output.binding_byte_offset == 0,
              "ACLNN MatMul reference allocations must start aligned");

      switch (plan.a.data_type) {
        case FLAGDNN_DATA_FLOAT32:
          ++type_counts[0];
          break;
        case FLAGDNN_DATA_FLOAT16:
          ++type_counts[1];
          break;
        case FLAGDNN_DATA_BFLOAT16:
          ++type_counts[2];
          break;
        case FLAGDNN_DATA_BOOLEAN:
        case FLAGDNN_DATA_FP8_E4M3:
        case FLAGDNN_DATA_FP8_E5M2:
          throw std::runtime_error("MatMul catalog contains an invalid dtype");
      }
      saw_matrix = saw_matrix || plan.output.dimensions.size() == 2;
      saw_batched = saw_batched || plan.output.dimensions.size() == 3;
      saw_multidimensional_broadcast =
          saw_multidimensional_broadcast ||
          (plan.a.dimensions.size() == 4 &&
           plan.b.dimensions.size() == 3 &&
           plan.output.dimensions ==
               std::vector<std::int64_t>({2, 3, 17, 23}));
      const bool strided =
          tensor_io::storage_element_count(plan.a) >
              tensor_io::element_count(plan.a) ||
          tensor_io::storage_element_count(plan.b) >
              tensor_io::element_count(plan.b) ||
          tensor_io::storage_element_count(plan.output) >
              tensor_io::element_count(plan.output);
      strided_count += static_cast<std::size_t>(strided);
      autotune_count += static_cast<std::size_t>(test_case.autotune);
    }
    require(type_counts == std::array<std::size_t, 3>({9, 9, 9}),
            "MatMul functional dtype counts must be 9/9/9");
    require(strided_count == 3,
            "MatMul functional catalog must contain three gapped cases");
    require(autotune_count == 1,
            "MatMul functional catalog must contain one autotune case");
    require(saw_matrix && saw_batched && saw_multidimensional_broadcast,
            "MatMul functional catalog lost a required rank/broadcast case");

    const auto benchmark_cases =
        flagdnn::benchmarking::matmul_benchmark_cases();
    require(benchmark_cases.size() == 24,
            "common MatMul benchmark catalog must contain 24 cases");
    std::array<std::size_t, 3> benchmark_type_counts{};
    for (const auto& benchmark_case : benchmark_cases) {
      require(benchmark_case.operation ==
                  flagdnn::benchmarking::Operation::kMatmul &&
                  benchmark_case.output_count == 1 &&
                  benchmark_case.tensors.size() == 3,
              "MatMul benchmark catalog contains a foreign case");
      const auto benchmark_plan =
          flagdnn::benchmarking::plan_aclnn_matmul_benchmark(
              benchmark_case);
      require(benchmark_plan.a.uid == benchmark_case.tensors[0].uid &&
                  benchmark_plan.a.dimensions ==
                      benchmark_case.tensors[0].dimensions &&
                  benchmark_plan.a.strides ==
                      benchmark_case.tensors[0].strides &&
                  benchmark_plan.b.uid == benchmark_case.tensors[1].uid &&
                  benchmark_plan.b.dimensions ==
                      benchmark_case.tensors[1].dimensions &&
                  benchmark_plan.b.strides ==
                      benchmark_case.tensors[1].strides &&
                  benchmark_plan.output.uid ==
                      benchmark_case.tensors[2].uid &&
                  benchmark_plan.output.dimensions ==
                      benchmark_case.tensors[2].dimensions &&
                  benchmark_plan.output.strides ==
                      benchmark_case.tensors[2].strides &&
                  benchmark_plan.a.binding_byte_offset == 0 &&
                  benchmark_plan.b.binding_byte_offset == 0 &&
                  benchmark_plan.output.binding_byte_offset == 0 &&
                  benchmark_plan.cube_math_type == 0,
              "ACLNN MatMul benchmark plan changed the case contract");
      switch (benchmark_plan.a.data_type) {
        case FLAGDNN_DATA_FLOAT32:
          ++benchmark_type_counts[0];
          break;
        case FLAGDNN_DATA_FLOAT16:
          ++benchmark_type_counts[1];
          break;
        case FLAGDNN_DATA_BFLOAT16:
          ++benchmark_type_counts[2];
          break;
        default:
          throw std::runtime_error(
              "MatMul benchmark catalog contains an invalid dtype");
      }
    }
    require(benchmark_type_counts ==
                std::array<std::size_t, 3>({8, 8, 8}),
            "MatMul benchmark dtype counts must be 8/8/8");

    MatmulTestCase invalid = cases.front();
    invalid.b.data_type = FLAGDNN_DATA_BOOLEAN;
    require_failure(
        [&] { (void)flagdnn::testing::plan_aclnn_matmul(invalid); },
        "data type");
    invalid = cases.front();
    invalid.output.dimensions.back() += 1;
    require_failure(
        [&] { (void)flagdnn::testing::plan_aclnn_matmul(invalid); },
        "output shape");
    invalid = cases.front();
    invalid.a.strides = {1, 1, 1};
    require_failure(
        [&] { (void)flagdnn::testing::plan_aclnn_matmul(invalid); },
        "overlap");

    std::cout << "Ascend MatMul validation contract: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Ascend MatMul validation contract failed: "
              << error.what() << '\n';
    return 1;
  }
}
