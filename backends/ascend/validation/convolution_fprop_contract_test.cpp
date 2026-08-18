/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/convolution.hpp"
#include "common/cases.hpp"
#include "validation/benchmark/aclnn_convolution_provider.hpp"
#include "validation/functional/aclnn_convolution.hpp"
#include "validation/tensor_io.hpp"

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
using flagdnn::testing::AclnnConvolutionFpropPlan;
using flagdnn::testing::ConvolutionMode;
using flagdnn::testing::ConvolutionTestCase;
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

std::vector<std::int64_t> contiguous_strides(
    const std::vector<std::int64_t>& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0U; --axis) {
    result[axis - 1U] = stride;
    stride *= dimensions[axis - 1U];
  }
  return result;
}

template <typename Callback>
void require_failure(Callback&& callback, std::string_view needle) {
  try {
    callback();
  } catch (const std::exception& error) {
    require(std::string_view(error.what()).find(needle) !=
                std::string_view::npos,
            "convolution plan failure did not identify its contract");
    return;
  }
  throw std::runtime_error("invalid ACLNN convolution plan was accepted");
}

}  // namespace

int main() {
  try {
    const std::vector<ConvolutionTestCase> cases =
        flagdnn::testing::make_convolution_cases(
            flagdnn::testing::ConvolutionDirection::kFprop);
    require(cases.size() == 24U,
            "common convolution fprop catalog must contain 24 cases");
    std::array<std::size_t, 3> dtype_counts{};
    std::array<std::size_t, 3> rank_counts{};
    std::size_t explicit_padding_count = 0U;
    std::size_t grouped_count = 0U;
    bool saw_asymmetric_2d = false;
    bool saw_asymmetric_3d = false;
    for (const ConvolutionTestCase& test_case : cases) {
      const AclnnConvolutionFpropPlan plan =
          flagdnn::testing::plan_aclnn_convolution_fprop(test_case);
      require(same_tensor(plan.input, test_case.x) &&
                  same_tensor(plan.filter, test_case.w) &&
                  same_tensor(plan.output, test_case.y),
              "ACLNN convolution plan changed a public tensor contract");
      require(plan.input.binding_byte_offset == 0U &&
                  plan.filter.binding_byte_offset == 0U &&
                  plan.output.binding_byte_offset == 0U,
              "ACLNN convolution reference tensors must start aligned");
      require(plan.groups == test_case.groups &&
                  plan.stride == test_case.stride &&
                  plan.dilation == test_case.dilation &&
                  plan.output_padding ==
                      std::vector<std::int64_t>(test_case.stride.size(), 0) &&
                  !plan.transposed && plan.cube_math_type == 0,
              "ACLNN convolution scalar/vector parameters changed");
      const std::size_t rank = test_case.x.dimensions.size() - 2U;
      ++rank_counts[rank - 1U];
      switch (test_case.x.data_type) {
        case FLAGDNN_DATA_FLOAT32:
          ++dtype_counts[0];
          break;
        case FLAGDNN_DATA_FLOAT16:
          ++dtype_counts[1];
          break;
        case FLAGDNN_DATA_BFLOAT16:
          ++dtype_counts[2];
          break;
        default:
          throw std::runtime_error(
              "convolution catalog contains an invalid data type");
      }
      const bool asymmetric =
          test_case.pre_padding != test_case.post_padding;
      const bool requires_explicit_padding = asymmetric && rank == 3U;
      require(plan.requires_explicit_padding() == requires_explicit_padding,
              "ACLNN asymmetric padding strategy is inconsistent");
      if (!asymmetric) {
        require(plan.convolution_padding == test_case.pre_padding &&
                    same_tensor(plan.convolution_input, plan.input),
                "ACLNN symmetric convolution plan changed its input");
      } else if (!requires_explicit_padding) {
        std::vector<std::int64_t> native_padding;
        native_padding.reserve(rank * 2U);
        for (std::size_t axis = 0U; axis < rank; ++axis) {
          native_padding.push_back(test_case.pre_padding[axis]);
          native_padding.push_back(test_case.post_padding[axis]);
        }
        require(plan.explicit_padding.empty() &&
                    plan.convolution_padding == native_padding &&
                    same_tensor(plan.convolution_input, plan.input),
                "ACLNN native asymmetric padding order is invalid");
        saw_asymmetric_2d = saw_asymmetric_2d || rank == 2U;
      } else {
        ++explicit_padding_count;
        std::vector<std::int64_t> expected_padding;
        for (std::size_t axis = rank; axis != 0U; --axis) {
          expected_padding.push_back(test_case.pre_padding[axis - 1U]);
          expected_padding.push_back(test_case.post_padding[axis - 1U]);
        }
        require(plan.explicit_padding == expected_padding &&
                    plan.convolution_padding ==
                        std::vector<std::int64_t>(rank, 0),
                "ACLNN explicit padding order is invalid");
        std::vector<std::int64_t> padded = test_case.x.dimensions;
        for (std::size_t axis = 0U; axis < rank; ++axis) {
          padded[axis + 2U] += test_case.pre_padding[axis] +
                               test_case.post_padding[axis];
        }
        require(plan.convolution_input.dimensions == padded &&
                    plan.convolution_input.strides ==
                        contiguous_strides(padded) &&
                    plan.convolution_input.data_type ==
                        test_case.x.data_type,
                "ACLNN explicitly padded tensor contract is invalid");
        saw_asymmetric_3d = saw_asymmetric_3d || rank == 3U;
      }
      grouped_count += static_cast<std::size_t>(test_case.groups != 1);
      (void)tensor_io::encoded_byte_count(plan.convolution_input);
    }
    require(dtype_counts == std::array<std::size_t, 3>({8U, 8U, 8U}),
            "convolution functional dtype counts must be 8/8/8");
    require(rank_counts == std::array<std::size_t, 3>({3U, 15U, 6U}),
            "convolution functional rank counts must be 3/15/6");
    require(explicit_padding_count == 3U && saw_asymmetric_2d &&
                saw_asymmetric_3d,
            "convolution asymmetric padding coverage is incomplete");
    require(grouped_count == 3U,
            "convolution grouped coverage must contain three dtype cases");

    std::unique_ptr<flagdnn::testing::ConvolutionExecutable> executable =
        flagdnn::testing::build_convolution_reference(cases.front());
    require(executable != nullptr && executable->workspace_size() == 0U,
            "ACLNN convolution reference construction is invalid");

    const std::vector<flagdnn::benchmarking::BenchmarkCase>
        correctness_cases = flagdnn::benchmarking::conv_fprop_cases();
    const std::vector<flagdnn::benchmarking::BenchmarkCase> benchmark_cases =
        flagdnn::benchmarking::conv_fprop_benchmark_cases();
    require(correctness_cases.size() == 34U && benchmark_cases.size() == 51U,
            "common convolution benchmark catalogs must contain 34/51 cases");
    std::array<std::size_t, 3> benchmark_dtype_counts{};
    for (const flagdnn::benchmarking::BenchmarkCase& specification :
         benchmark_cases) {
      const flagdnn::benchmarking::AclnnConvolutionFpropBenchmarkPlan plan =
          flagdnn::benchmarking::plan_aclnn_convolution_fprop_benchmark(
              specification);
      require(plan.input.uid == specification.tensors[0].uid &&
                  plan.filter.uid == specification.tensors[1].uid &&
                  plan.output.uid == specification.tensors[2].uid &&
                  plan.input.binding_byte_offset == 0U &&
                  plan.filter.binding_byte_offset == 0U &&
                  plan.output.binding_byte_offset == 0U &&
                  plan.groups == specification.convolution.groups &&
                  plan.stride == specification.convolution.stride &&
                  plan.dilation == specification.convolution.dilation,
              "ACLNN convolution benchmark plan changed the public contract");
      switch (plan.input.data_type) {
        case FLAGDNN_DATA_FLOAT32:
          ++benchmark_dtype_counts[0];
          break;
        case FLAGDNN_DATA_FLOAT16:
          ++benchmark_dtype_counts[1];
          break;
        case FLAGDNN_DATA_BFLOAT16:
          ++benchmark_dtype_counts[2];
          break;
        default:
          throw std::runtime_error(
              "convolution benchmark contains an invalid dtype");
      }
    }
    require(benchmark_dtype_counts ==
                std::array<std::size_t, 3>({17U, 17U, 17U}),
            "convolution benchmark dtype counts must be 17/17/17");

    ConvolutionTestCase invalid = cases.front();
    invalid.mode = ConvolutionMode::kConvolution;
    require_failure(
        [&] {
          (void)flagdnn::testing::plan_aclnn_convolution_fprop(invalid);
        },
        "cross-correlation");
    invalid = cases.front();
    invalid.y.dimensions.back() += 1;
    require_failure(
        [&] {
          (void)flagdnn::testing::plan_aclnn_convolution_fprop(invalid);
        },
        "output");

    std::cout << "Ascend convolution fprop validation contract: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Ascend convolution fprop validation contract failed: "
              << error.what() << '\n';
    return 1;
  }
}
