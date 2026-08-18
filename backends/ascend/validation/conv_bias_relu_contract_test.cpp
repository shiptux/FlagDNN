/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/composite.hpp"
#include "validation/functional/aclnn_conv_bias_relu.hpp"
#include "validation/functional/aclnn_convolution.hpp"

#include <acl/acl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

using flagdnn::testing::AclnnConvBiasReluPlan;
using flagdnn::testing::ConvBiasReluTestCase;
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
            "ConvBiasRelu failure did not identify its contract");
    return;
  }
  throw std::runtime_error("invalid ACLNN ConvBiasRelu plan was accepted");
}

}  // namespace

int main() {
  try {
    const std::vector<ConvBiasReluTestCase> cases =
        flagdnn::testing::make_conv_bias_relu_cases();
    require(cases.size() == 30U,
            "common ConvBiasRelu catalog must contain 30 cases");

    std::array<std::size_t, 3> data_type_counts{};
    std::size_t autotune_count = 0U;
    bool saw_reference_layout_conversion = false;
    for (const ConvBiasReluTestCase& test_case : cases) {
      const AclnnConvBiasReluPlan plan =
          flagdnn::testing::plan_aclnn_conv_bias_relu(test_case);
      require(same_tensor(plan.x, test_case.x) &&
                  same_tensor(plan.w, test_case.w) &&
                  same_tensor(plan.bias, test_case.bias),
              "ACLNN ConvBiasRelu plan changed an input tensor");
      require(plan.convolution.uid == test_case.output.uid + 1 &&
                  plan.biased.uid == test_case.output.uid + 2,
              "ACLNN ConvBiasRelu intermediate UIDs are unstable");
      const std::vector<std::int64_t> contiguous =
          contiguous_strides(test_case.output.dimensions);
      require(plan.convolution.data_type == test_case.output.data_type &&
                  plan.convolution.dimensions ==
                      test_case.output.dimensions &&
                  plan.convolution.strides == test_case.output.strides &&
                  plan.convolution.binding_byte_offset == 0U &&
                  plan.biased.data_type == test_case.output.data_type &&
                  plan.biased.dimensions == test_case.output.dimensions &&
                  plan.biased.strides == contiguous &&
                  plan.biased.binding_byte_offset == 0U &&
                  plan.output.uid == test_case.output.uid &&
                  plan.output.data_type == test_case.output.data_type &&
                  plan.output.dimensions == test_case.output.dimensions &&
                  plan.output.strides == contiguous &&
                  plan.output.binding_byte_offset == 0U,
              "ACLNN ConvBiasRelu intermediate layout is invalid");
      const std::unordered_set<std::int64_t> uids = {
          plan.x.uid,
          plan.w.uid,
          plan.bias.uid,
          plan.convolution.uid,
          plan.biased.uid,
          plan.output.uid,
      };
      require(uids.size() == 6U,
              "ACLNN ConvBiasRelu plan contains an aliased UID");
      saw_reference_layout_conversion =
          saw_reference_layout_conversion ||
          plan.convolution.strides != plan.biased.strides;
      switch (test_case.x.data_type) {
        case FLAGDNN_DATA_FLOAT32:
          ++data_type_counts[0];
          break;
        case FLAGDNN_DATA_FLOAT16:
          ++data_type_counts[1];
          break;
        case FLAGDNN_DATA_BFLOAT16:
          ++data_type_counts[2];
          break;
        default:
          throw std::runtime_error(
              "ConvBiasRelu catalog contains an invalid data type");
      }
      autotune_count += static_cast<std::size_t>(test_case.autotune);
    }
    require(data_type_counts ==
                std::array<std::size_t, 3>({10U, 10U, 10U}),
            "ConvBiasRelu functional dtype counts must be 10/10/10");
    require(autotune_count == 1U,
            "ConvBiasRelu catalog must contain one autotune case");
    require(saw_reference_layout_conversion,
            "ConvBiasRelu contract never exercises provider layouts");

    std::unique_ptr<flagdnn::testing::CompositeExecutable> executable =
        flagdnn::testing::build_conv_bias_relu_reference(cases.front());
    require(executable != nullptr && executable->workspace_size() == 0U,
            "ACLNN ConvBiasRelu reference construction is invalid");

    ConvBiasReluTestCase invalid = cases.front();
    invalid.x.uid = invalid.output.uid + 1;
    require_failure(
        [&] {
          (void)flagdnn::testing::plan_aclnn_conv_bias_relu(invalid);
        },
        "collides");

    invalid = cases.front();
    invalid.output.uid = std::numeric_limits<std::int64_t>::max() - 1;
    require_failure(
        [&] {
          (void)flagdnn::testing::plan_aclnn_conv_bias_relu(invalid);
        },
        "overflow");

    invalid = cases.front();
    invalid.bias.uid = invalid.x.uid;
    require_failure(
        [&] {
          (void)flagdnn::testing::plan_aclnn_conv_bias_relu(invalid);
        },
        "metadata");

    require(flagdnn::testing::aclnn_convolution_status_is_unsupported(
                ACL_ERROR_UNSUPPORTED_DATA_TYPE, "unsupported dtype"),
            "known ACLNN convolution unsupported status was not classified");
    require(flagdnn::testing::aclnn_convolution_status_is_unsupported(
                1234567, "operator is not supported for this layout"),
            "explicit ACLNN convolution unsupported message was not classified");
    require(!flagdnn::testing::aclnn_convolution_status_is_unsupported(
                1234567, "invalid convolution descriptor"),
            "unexpected ACLNN convolution failure was classified as skip");
    const flagdnn::testing::AclnnConvolutionUnsupportedError unsupported(
        ACL_ERROR_UNSUPPORTED_DATA_TYPE, "unsupported dtype");
    require(unsupported.status() == ACL_ERROR_UNSUPPORTED_DATA_TYPE,
            "typed ACLNN convolution unsupported status was not preserved");

    std::cout << "Ascend ConvBiasRelu validation contract: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Ascend ConvBiasRelu validation contract failed: "
              << error.what() << '\n';
    return 1;
  }
}
