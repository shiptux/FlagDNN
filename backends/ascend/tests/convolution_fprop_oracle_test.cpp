/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/convolution_fprop_oracle.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using flagdnn::ascend::ArgumentSource;
using flagdnn::ascend::ArgumentSourceKind;
using flagdnn::ascend::AscendStageArtifact;
using flagdnn::ascend::KernelFamily;
using flagdnn::ascend::RawArgumentType;
using flagdnn::ascend::StorageDataType;

std::size_t element_size(StorageDataType type) {
  return type == StorageDataType::kFloat32 ? sizeof(float)
                                           : sizeof(std::uint16_t);
}

std::uint16_t float_to_half(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t exponent = (bits >> 23U) & 0xffU;
  const std::uint32_t fraction = bits & 0x007fffffU;
  if (exponent == 0xffU) {
    return static_cast<std::uint16_t>(
        sign | 0x7c00U | (fraction == 0U ? 0U : 0x0200U));
  }
  const std::int32_t half_exponent = static_cast<std::int32_t>(exponent) - 112;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t mantissa = fraction | 0x00800000U;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - half_exponent);
    const std::uint32_t rounded = mantissa + ((1U << (shift - 1U)) - 1U) +
                                  ((mantissa >> shift) & 1U);
    return static_cast<std::uint16_t>(sign | (rounded >> shift));
  }
  const std::uint32_t rounded = fraction + 0x00000fffU +
                                ((fraction >> 13U) & 1U);
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(half_exponent) << 10U) |
      (rounded >> 13U));
}

float half_to_float(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  const std::uint32_t exponent = (value >> 10U) & 0x1fU;
  std::uint32_t fraction = value & 0x03ffU;
  std::uint32_t bits = 0;
  if (exponent == 0U) {
    if (fraction == 0U) {
      bits = sign;
    } else {
      std::uint32_t normalized_exponent = 113U;
      while ((fraction & 0x0400U) == 0U) {
        fraction <<= 1U;
        --normalized_exponent;
      }
      bits = sign | (normalized_exponent << 23U) |
             ((fraction & 0x03ffU) << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (fraction << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
  }
  return std::bit_cast<float>(bits);
}

void store_value(std::span<std::uint8_t> bytes,
                 std::size_t element,
                 StorageDataType type,
                 float value) {
  const std::size_t size = element_size(type);
  if (element >= bytes.size() / size) {
    throw std::runtime_error("test store offset is out of range");
  }
  if (type == StorageDataType::kFloat32) {
    std::memcpy(bytes.data() + element * size, &value, size);
    return;
  }
  std::uint16_t encoded = 0;
  if (type == StorageDataType::kFloat16) {
    encoded = float_to_half(value);
  } else {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x00007fffU + ((bits >> 16U) & 1U);
    encoded = static_cast<std::uint16_t>(bits >> 16U);
  }
  std::memcpy(bytes.data() + element * size, &encoded, size);
}

float load_value(std::span<const std::uint8_t> bytes,
                 std::size_t element,
                 StorageDataType type) {
  const std::size_t size = element_size(type);
  if (element >= bytes.size() / size) {
    throw std::runtime_error("test load offset is out of range");
  }
  if (type == StorageDataType::kFloat32) {
    float result = 0.0F;
    std::memcpy(&result, bytes.data() + element * size, size);
    return result;
  }
  std::uint16_t encoded = 0;
  std::memcpy(&encoded, bytes.data() + element * size, size);
  if (type == StorageDataType::kFloat16) {
    return half_to_float(encoded);
  }
  return std::bit_cast<float>(static_cast<std::uint32_t>(encoded) << 16U);
}

std::size_t storage_elements(const std::array<std::int64_t, 5>& dimensions,
                             const std::array<std::int64_t, 5>& strides) {
  std::size_t maximum = 0U;
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    maximum += static_cast<std::size_t>(dimensions[axis] - 1) *
               static_cast<std::size_t>(strides[axis]);
  }
  return maximum + 1U;
}

std::size_t tensor_offset(const std::array<std::int64_t, 5>& coordinates,
                          const std::array<std::int64_t, 5>& strides) {
  std::size_t result = 0U;
  for (std::size_t axis = 0; axis < coordinates.size(); ++axis) {
    result += static_cast<std::size_t>(coordinates[axis] * strides[axis]);
  }
  return result;
}

ArgumentSource pointer_argument(std::size_t index,
                                std::string name,
                                std::int64_t uid,
                                std::size_t size) {
  ArgumentSource result;
  result.index = index;
  result.name = std::move(name);
  result.source = ArgumentSourceKind::kBinding;
  result.type = RawArgumentType::kPointer;
  result.uid = uid;
  result.size = size;
  result.alignment = 16U;
  return result;
}

AscendStageArtifact make_stage(StorageDataType type, int spatial_rank) {
  AscendStageArtifact stage;
  stage.operation = "convolution_fprop";
  stage.kernel_family = KernelFamily::kConvolutionFprop;
  stage.tensor_storage_data_types = {type, type, type};
  stage.input_count = 2U;
  stage.convolution_spatial_rank = spatial_rank;
  stage.convolution_groups = spatial_rank == 3 ? 1 : 2;
  stage.convolution_pre_padding.fill(0);
  stage.convolution_post_padding.fill(0);
  stage.convolution_stride.fill(1);
  stage.convolution_dilation.fill(1);
  if (spatial_rank == 1) {
    stage.convolution_input_dimensions = {1, 4, 1, 1, 5};
    stage.convolution_input_strides = {80, 18, 0, 0, 2};
    stage.convolution_filter_dimensions = {4, 2, 1, 1, 3};
    stage.convolution_filter_strides = {100, 20, 0, 0, 2};
    stage.convolution_output_dimensions = {1, 4, 1, 1, 5};
    stage.convolution_output_strides = {100, 20, 0, 0, 2};
    stage.convolution_pre_padding = {0, 0, 1};
    stage.convolution_post_padding = {0, 0, 1};
  } else if (spatial_rank == 2) {
    stage.convolution_input_dimensions = {1, 4, 1, 5, 6};
    stage.convolution_input_strides = {300, 70, 0, 9, 1};
    stage.convolution_filter_dimensions = {6, 2, 1, 3, 2};
    stage.convolution_filter_strides = {100, 30, 0, 8, 1};
    stage.convolution_output_dimensions = {1, 6, 1, 4, 3};
    stage.convolution_output_strides = {400, 60, 0, 10, 2};
    stage.convolution_pre_padding = {0, 1, 0};
    stage.convolution_post_padding = {0, 0, 1};
    stage.convolution_stride = {1, 1, 2};
    stage.convolution_dilation = {1, 1, 2};
  } else {
    stage.convolution_input_dimensions = {1, 2, 4, 4, 4};
    stage.convolution_input_strides = {128, 1, 32, 8, 2};
    stage.convolution_filter_dimensions = {3, 2, 2, 2, 2};
    stage.convolution_filter_strides = {16, 1, 8, 4, 2};
    stage.convolution_output_dimensions = {1, 3, 3, 3, 3};
    stage.convolution_output_strides = {81, 1, 27, 9, 3};
  }
  stage.convolution_input_channels = stage.convolution_input_dimensions[1];
  stage.convolution_output_channels = stage.convolution_filter_dimensions[0];
  stage.convolution_channels_per_group =
      stage.convolution_input_channels / stage.convolution_groups;
  stage.n_elements = 1;
  for (const std::int64_t dimension : stage.convolution_output_dimensions) {
    stage.n_elements *= static_cast<std::int32_t>(dimension);
  }
  const std::size_t size = element_size(type);
  stage.arguments = {
      pointer_argument(
          0U,
          "input_ptr",
          1,
          storage_elements(stage.convolution_input_dimensions,
                           stage.convolution_input_strides) *
              size),
      pointer_argument(
          1U,
          "filter_ptr",
          2,
          storage_elements(stage.convolution_filter_dimensions,
                           stage.convolution_filter_strides) *
              size),
      pointer_argument(
          2U,
          "output_ptr",
          3,
          storage_elements(stage.convolution_output_dimensions,
                           stage.convolution_output_strides) *
              size),
  };
  ArgumentSource scalar;
  scalar.index = 3U;
  scalar.name = "n_elements";
  scalar.source = ArgumentSourceKind::kScalar;
  scalar.type = RawArgumentType::kI32;
  scalar.scalar = stage.n_elements;
  stage.arguments.push_back(std::move(scalar));
  return stage;
}

void expect_invalid(const std::function<void()>& action,
                    const char* message) {
  try {
    action();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

double manual_first_output(const AscendStageArtifact& stage,
                           std::span<const std::uint8_t> input,
                           std::span<const std::uint8_t> filter) {
  double result = 0.0;
  for (std::int64_t channel = 0;
       channel < stage.convolution_channels_per_group;
       ++channel) {
    for (std::int64_t d = 0;
         d < stage.convolution_filter_dimensions[2];
         ++d) {
      for (std::int64_t h = 0;
           h < stage.convolution_filter_dimensions[3];
           ++h) {
        for (std::int64_t w = 0;
             w < stage.convolution_filter_dimensions[4];
             ++w) {
          const std::array<std::int64_t, 3> input_spatial = {
              d * stage.convolution_dilation[0] -
                  stage.convolution_pre_padding[0],
              h * stage.convolution_dilation[1] -
                  stage.convolution_pre_padding[1],
              w * stage.convolution_dilation[2] -
                  stage.convolution_pre_padding[2]};
          if (input_spatial[0] < 0 || input_spatial[1] < 0 ||
              input_spatial[2] < 0 ||
              input_spatial[0] >= stage.convolution_input_dimensions[2] ||
              input_spatial[1] >= stage.convolution_input_dimensions[3] ||
              input_spatial[2] >= stage.convolution_input_dimensions[4]) {
            continue;
          }
          const std::size_t input_offset = tensor_offset(
              {0, channel, input_spatial[0], input_spatial[1], input_spatial[2]},
              stage.convolution_input_strides);
          const std::size_t filter_offset = tensor_offset(
              {0, channel, d, h, w}, stage.convolution_filter_strides);
          result += static_cast<double>(
                        load_value(input, input_offset, stage.input_type(0))) *
                    static_cast<double>(
                        load_value(filter, filter_offset, stage.input_type(1)));
        }
      }
    }
  }
  return result;
}

std::set<std::size_t> output_offsets(const AscendStageArtifact& stage) {
  std::set<std::size_t> result;
  for (std::int64_t n = 0; n < stage.convolution_output_dimensions[0]; ++n) {
    for (std::int64_t k = 0; k < stage.convolution_output_dimensions[1]; ++k) {
      for (std::int64_t d = 0; d < stage.convolution_output_dimensions[2]; ++d) {
        for (std::int64_t h = 0; h < stage.convolution_output_dimensions[3]; ++h) {
          for (std::int64_t w = 0; w < stage.convolution_output_dimensions[4];
               ++w) {
            result.insert(tensor_offset(
                {n, k, d, h, w}, stage.convolution_output_strides));
          }
        }
      }
    }
  }
  return result;
}

void check_case(StorageDataType type, int spatial_rank) {
  AscendStageArtifact stage = make_stage(type, spatial_rank);
  flagdnn::ascend::validate_convolution_fprop_stage_runtime_contract(stage);
  std::vector<std::uint8_t> input(stage.arguments[0].size, 0xA5U);
  std::vector<std::uint8_t> filter(stage.arguments[1].size, 0xA5U);
  flagdnn::ascend::seed_convolution_fprop_host_inputs(stage, input, filter);
  const std::vector<std::uint8_t> seeded_input = input;
  const std::vector<std::uint8_t> seeded_filter = filter;
  std::fill(input.begin(), input.end(), 0xA5U);
  std::fill(filter.begin(), filter.end(), 0xA5U);
  flagdnn::ascend::seed_convolution_fprop_host_inputs(stage, input, filter);
  if (input != seeded_input || filter != seeded_filter) {
    throw std::runtime_error("convolution seed is not byte reproducible");
  }
  std::vector<std::uint8_t> output(stage.arguments[2].size, 0xA5U);
  flagdnn::ascend::compute_convolution_fprop_host_oracle(
      stage, input, filter, output);
  if (input != seeded_input || filter != seeded_filter) {
    throw std::runtime_error("convolution host oracle changed an input");
  }
  const double expected = manual_first_output(stage, input, filter);
  const float observed = load_value(output, 0U, type);
  const float tolerance = type == StorageDataType::kFloat32 ? 1.0e-4F : 0.5F;
  if (std::abs(observed - static_cast<float>(expected)) >
      tolerance * std::max(1.0F, std::abs(static_cast<float>(expected)))) {
    throw std::runtime_error("convolution oracle differs from manual sum");
  }
  flagdnn::ascend::validate_convolution_fprop_host_output(
      stage, input, filter, output);
  const std::set<std::size_t> active = output_offsets(stage);
  const std::size_t size = element_size(type);
  std::size_t padding = output.size() / size;
  for (std::size_t element = 0; element < output.size() / size; ++element) {
    if (!active.contains(element)) {
      padding = element;
      break;
    }
  }
  if (padding != output.size() / size &&
      !std::all_of(output.begin() + static_cast<std::ptrdiff_t>(padding * size),
                   output.begin() +
                       static_cast<std::ptrdiff_t>((padding + 1U) * size),
                   [](std::uint8_t byte) { return byte == 0xA5U; })) {
    throw std::runtime_error("convolution oracle overwrote output padding");
  }
  std::vector<std::uint8_t> corrupted = output;
  store_value(corrupted, 0U, type, observed + 64.0F);
  expect_invalid(
      [&] {
        flagdnn::ascend::validate_convolution_fprop_host_output(
            stage, input, filter, corrupted);
      },
      "convolution logical output corruption was accepted");
  if (padding != output.size() / size) {
    corrupted = output;
    corrupted[padding * size] ^= 1U;
    expect_invalid(
        [&] {
          flagdnn::ascend::validate_convolution_fprop_host_output(
              stage, input, filter, corrupted);
        },
        "convolution padding corruption was accepted");
  }
  std::vector<std::uint8_t> candidate = output;
  store_value(candidate, 0U, type, observed + 1.0e-5F);
  flagdnn::ascend::validate_convolution_fprop_candidate_outputs(
      stage, output, candidate);
  std::vector<std::uint8_t> shadow(output.size(), 0U);
  flagdnn::ascend::commit_convolution_fprop_host_output(stage, output, shadow);
  if (shadow != output) {
    throw std::runtime_error("convolution output shadow was not committed");
  }
}

void check_runtime_and_rejections() {
  if (flagdnn::ascend::convolution_fprop_kernel_input_count() != 2U ||
      flagdnn::ascend::convolution_fprop_output_argument_index() != 2U ||
      flagdnn::ascend::convolution_fprop_tensor_slot_count() != 3U ||
      flagdnn::ascend::convolution_fprop_runtime_argument_count() != 4U ||
      std::string(flagdnn::ascend::convolution_fprop_output_argument_name()) !=
          "output_ptr") {
    throw std::runtime_error("convolution runtime routing constants are invalid");
  }
  AscendStageArtifact stage = make_stage(StorageDataType::kFloat32, 2);
  flagdnn::ascend::validate_convolution_fprop_stage_runtime_contract(stage);
  AscendStageArtifact changed = stage;
  changed.arguments[1].name = "weights_ptr";
  expect_invalid(
      [&] {
        flagdnn::ascend::validate_convolution_fprop_stage_runtime_contract(
            changed);
      },
      "convolution ABI name mutation was accepted");
  changed = stage;
  changed.arguments[3].type = RawArgumentType::kI64;
  expect_invalid(
      [&] {
        flagdnn::ascend::validate_convolution_fprop_stage_runtime_contract(
            changed);
      },
      "convolution scalar ABI type mutation was accepted");
  changed = stage;
  changed.convolution_groups = 3;
  expect_invalid(
      [&] {
        flagdnn::ascend::validate_convolution_fprop_stage_runtime_contract(
            changed);
      },
      "convolution invalid groups were accepted");
  changed = stage;
  changed.convolution_output_strides[4] = 0;
  expect_invalid(
      [&] {
        flagdnn::ascend::validate_convolution_fprop_stage_runtime_contract(
            changed);
      },
      "convolution overlapping output strides were accepted");

  std::vector<std::uint8_t> input(stage.arguments[0].size, 0xA5U);
  std::vector<std::uint8_t> filter(stage.arguments[1].size, 0xA5U);
  std::vector<std::uint8_t> output(stage.arguments[2].size, 0xA5U);
  flagdnn::ascend::seed_convolution_fprop_host_inputs(stage, input, filter);
  expect_invalid(
      [&] {
        flagdnn::ascend::compute_convolution_fprop_host_oracle(
            stage,
            std::span<const std::uint8_t>(input).first(input.size() - 1U),
            filter,
            output);
      },
      "convolution short input storage was accepted");
  expect_invalid(
      [&] {
        std::vector<std::uint8_t> aliased_storage(
            std::max(stage.arguments[0].size, stage.arguments[2].size),
            0xA5U);
        flagdnn::ascend::compute_convolution_fprop_host_oracle(
            stage,
            std::span<const std::uint8_t>(aliased_storage)
                .first(stage.arguments[0].size),
            filter,
            std::span<std::uint8_t>(aliased_storage)
                .first(stage.arguments[2].size));
      },
      "convolution output/input alias was accepted");
  store_value(input,
              0U,
              StorageDataType::kFloat32,
              std::numeric_limits<float>::infinity());
  expect_invalid(
      [&] {
        flagdnn::ascend::compute_convolution_fprop_host_oracle(
            stage, input, filter, output);
      },
      "convolution nonfinite input was accepted");
}

}  // namespace

int main() {
  try {
    check_case(StorageDataType::kFloat32, 1);
    check_case(StorageDataType::kFloat16, 2);
    check_case(StorageDataType::kBFloat16, 3);
    check_runtime_and_rejections();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
