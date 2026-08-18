/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/convolution_fprop_oracle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace flagdnn::ascend {
namespace {

constexpr std::uint64_t kFullOracleOutputLimit = 65536U;
constexpr std::uint64_t kLargeOracleSampleCount = 257U;
constexpr std::uint8_t kPaddingByte = 0xA5U;

[[noreturn]] void invalid_oracle(const char* message) {
  throw std::invalid_argument(message);
}

std::size_t storage_element_size(StorageDataType type) {
  switch (type) {
    case StorageDataType::kFloat32:
      return sizeof(float);
    case StorageDataType::kFloat16:
    case StorageDataType::kBFloat16:
      return sizeof(std::uint16_t);
    case StorageDataType::kBoolean:
      break;
  }
  invalid_oracle("convolution storage data type is invalid");
}

std::uint64_t checked_add(std::uint64_t left,
                          std::uint64_t right,
                          const char* message) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    invalid_oracle(message);
  }
  return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left,
                               std::uint64_t right,
                               const char* message) {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    invalid_oracle(message);
  }
  return left * right;
}

std::size_t checked_size(std::uint64_t value, const char* message) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    invalid_oracle(message);
  }
  return static_cast<std::size_t>(value);
}

std::uint16_t float_to_half(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
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
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float load_value(std::span<const std::uint8_t> bytes,
                 StorageDataType type,
                 std::uint64_t element) {
  const std::size_t size = storage_element_size(type);
  if (element >= bytes.size() / size) {
    invalid_oracle("convolution tensor offset is out of range");
  }
  float result = 0.0F;
  if (type == StorageDataType::kFloat32) {
    std::memcpy(&result, bytes.data() + element * size, sizeof(result));
  } else {
    std::uint16_t encoded = 0;
    std::memcpy(&encoded, bytes.data() + element * size, sizeof(encoded));
    if (type == StorageDataType::kFloat16) {
      result = half_to_float(encoded);
    } else {
      const std::uint32_t bits = static_cast<std::uint32_t>(encoded) << 16U;
      std::memcpy(&result, &bits, sizeof(result));
    }
  }
  if (!std::isfinite(result)) {
    invalid_oracle("convolution values must be finite");
  }
  return result;
}

void store_value(std::span<std::uint8_t> bytes,
                 StorageDataType type,
                 std::uint64_t element,
                 float value) {
  if (!std::isfinite(value)) {
    invalid_oracle("convolution result must be finite");
  }
  const std::size_t size = storage_element_size(type);
  if (element >= bytes.size() / size) {
    invalid_oracle("convolution tensor offset is out of range");
  }
  if (type == StorageDataType::kFloat32) {
    std::memcpy(bytes.data() + element * size, &value, sizeof(value));
    return;
  }
  std::uint16_t encoded = 0;
  if (type == StorageDataType::kFloat16) {
    encoded = float_to_half(value);
    if ((encoded & 0x7c00U) == 0x7c00U) {
      invalid_oracle("convolution result must be finite");
    }
  } else {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x00007fffU + ((bits >> 16U) & 1U);
    encoded = static_cast<std::uint16_t>(bits >> 16U);
    if ((encoded & 0x7f80U) == 0x7f80U) {
      invalid_oracle("convolution result must be finite");
    }
  }
  std::memcpy(bytes.data() + element * size, &encoded, sizeof(encoded));
}

void validate_non_overlapping_tensor(
    const std::array<std::int64_t, 5>& dimensions,
    const std::array<std::int64_t, 5>& strides,
    const char* message) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> active;
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    if (dimensions[axis] <= 0 || strides[axis] < 0) {
      invalid_oracle(message);
    }
    if (dimensions[axis] > 1) {
      if (strides[axis] == 0) {
        invalid_oracle(message);
      }
      active.emplace_back(static_cast<std::uint64_t>(strides[axis]),
                          static_cast<std::uint64_t>(dimensions[axis]));
    }
  }
  std::sort(active.begin(), active.end());
  std::uint64_t span = 1U;
  for (const auto& [stride, dimension] : active) {
    if (stride < span) {
      invalid_oracle(message);
    }
    span = checked_add(
        span,
        checked_multiply(dimension - 1U, stride, message),
        message);
  }
}

std::uint64_t storage_elements(
    const std::array<std::int64_t, 5>& dimensions,
    const std::array<std::int64_t, 5>& strides) {
  std::uint64_t maximum = 0U;
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    if (dimensions[axis] <= 0 || strides[axis] < 0) {
      invalid_oracle("convolution tensor metadata is invalid");
    }
    maximum = checked_add(
        maximum,
        checked_multiply(static_cast<std::uint64_t>(dimensions[axis] - 1),
                         static_cast<std::uint64_t>(strides[axis]),
                         "convolution storage span overflows"),
        "convolution storage span overflows");
  }
  return checked_add(maximum, 1U, "convolution storage span overflows");
}

std::size_t storage_bytes(const std::array<std::int64_t, 5>& dimensions,
                          const std::array<std::int64_t, 5>& strides,
                          StorageDataType type) {
  return checked_size(
      checked_multiply(storage_elements(dimensions, strides),
                       static_cast<std::uint64_t>(storage_element_size(type)),
                       "convolution storage byte count overflows"),
      "convolution storage byte count exceeds size_t");
}

std::uint64_t tensor_offset(
    const std::array<std::uint64_t, 5>& coordinates,
    const std::array<std::int64_t, 5>& strides) {
  std::uint64_t result = 0U;
  for (std::size_t axis = 0; axis < coordinates.size(); ++axis) {
    result = checked_add(
        result,
        checked_multiply(coordinates[axis],
                         static_cast<std::uint64_t>(strides[axis]),
                         "convolution tensor offset overflows"),
        "convolution tensor offset overflows");
  }
  return result;
}

std::array<std::uint64_t, 5> output_coordinates(
    const AscendStageArtifact& stage,
    std::uint64_t logical) {
  std::array<std::uint64_t, 5> coordinates{};
  for (std::size_t reversed = 5U; reversed != 0U; --reversed) {
    const std::size_t axis = reversed - 1U;
    const auto dimension = static_cast<std::uint64_t>(
        stage.convolution_output_dimensions[axis]);
    coordinates[axis] = logical % dimension;
    logical /= dimension;
  }
  if (logical != 0U) {
    invalid_oracle("convolution logical output index exceeds shape");
  }
  return coordinates;
}

bool spans_overlap(std::span<const std::uint8_t> left,
                   std::span<const std::uint8_t> right) {
  if (left.empty() || right.empty()) {
    return false;
  }
  const auto left_begin = reinterpret_cast<std::uintptr_t>(left.data());
  const auto right_begin = reinterpret_cast<std::uintptr_t>(right.data());
  if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left.size() ||
      right_begin >
          std::numeric_limits<std::uintptr_t>::max() - right.size()) {
    invalid_oracle("convolution host span address overflows");
  }
  const std::uintptr_t left_end = left_begin + left.size();
  const std::uintptr_t right_end = right_begin + right.size();
  return left_begin < right_end && right_begin < left_end;
}

void validate_host_spans(const AscendStageArtifact& stage,
                         std::span<const std::uint8_t> input,
                         std::span<const std::uint8_t> filter,
                         std::span<const std::uint8_t> output) {
  if (input.size() != stage.arguments[0].size ||
      filter.size() != stage.arguments[1].size ||
      output.size() != stage.arguments[2].size) {
    invalid_oracle("convolution host storage size is invalid");
  }
  if (spans_overlap(input, output) || spans_overlap(filter, output)) {
    invalid_oracle("convolution output aliases an input");
  }
}

bool valid_alignment(std::size_t alignment) {
  return alignment != 0U && (alignment & (alignment - 1U)) == 0U;
}

void validate_pointer_argument(const ArgumentSource& argument,
                               std::size_t index,
                               const char* name) {
  if (argument.index != index || argument.name != name ||
      argument.type != RawArgumentType::kPointer || argument.uid <= 0 ||
      (argument.source != ArgumentSourceKind::kBinding &&
       argument.source != ArgumentSourceKind::kGraphWorkspace) ||
      !valid_alignment(argument.alignment) ||
      (argument.source == ArgumentSourceKind::kBinding &&
       argument.workspace_offset != 0U) ||
      (argument.source == ArgumentSourceKind::kGraphWorkspace &&
       argument.workspace_offset % argument.alignment != 0U)) {
    invalid_oracle("convolution pointer ABI is invalid");
  }
}

std::uint64_t output_element_count(const AscendStageArtifact& stage) {
  std::uint64_t result = 1U;
  for (const std::int64_t dimension : stage.convolution_output_dimensions) {
    result = checked_multiply(result,
                              static_cast<std::uint64_t>(dimension),
                              "convolution output element count overflows");
  }
  return result;
}

std::uint64_t reduction_element_count(const AscendStageArtifact& stage) {
  std::uint64_t result =
      static_cast<std::uint64_t>(stage.convolution_channels_per_group);
  for (std::size_t axis = 0; axis < 3U; ++axis) {
    result = checked_multiply(
        result,
        static_cast<std::uint64_t>(stage.convolution_filter_dimensions[axis + 2U]),
        "convolution reduction element count overflows");
  }
  return result;
}

float expected_value(const AscendStageArtifact& stage,
                     std::span<const std::uint8_t> input,
                     std::span<const std::uint8_t> filter,
                     const std::array<std::uint64_t, 5>& output) {
  const std::uint64_t output_channels_per_group =
      static_cast<std::uint64_t>(stage.convolution_output_channels /
                                 stage.convolution_groups);
  const std::uint64_t group = output[1] / output_channels_per_group;
  const std::uint64_t input_channel_base =
      group * static_cast<std::uint64_t>(stage.convolution_channels_per_group);
  double accumulator = 0.0;
  for (std::uint64_t channel = 0;
       channel <
       static_cast<std::uint64_t>(stage.convolution_channels_per_group);
       ++channel) {
    for (std::uint64_t filter_d = 0;
         filter_d <
         static_cast<std::uint64_t>(stage.convolution_filter_dimensions[2]);
         ++filter_d) {
      for (std::uint64_t filter_h = 0;
           filter_h <
           static_cast<std::uint64_t>(stage.convolution_filter_dimensions[3]);
           ++filter_h) {
        for (std::uint64_t filter_w = 0;
             filter_w <
             static_cast<std::uint64_t>(stage.convolution_filter_dimensions[4]);
             ++filter_w) {
          const std::array<std::uint64_t, 3> filter_spatial = {
              filter_d, filter_h, filter_w};
          std::array<std::uint64_t, 3> input_spatial{};
          bool in_bounds = true;
          for (std::size_t axis = 0; axis < 3U; ++axis) {
            const std::uint64_t unpadded = checked_add(
                checked_multiply(
                    output[axis + 2U],
                    static_cast<std::uint64_t>(stage.convolution_stride[axis]),
                    "convolution input coordinate overflows"),
                checked_multiply(
                    filter_spatial[axis],
                    static_cast<std::uint64_t>(stage.convolution_dilation[axis]),
                    "convolution input coordinate overflows"),
                "convolution input coordinate overflows");
            const auto padding = static_cast<std::uint64_t>(
                stage.convolution_pre_padding[axis]);
            if (unpadded < padding) {
              in_bounds = false;
              break;
            }
            input_spatial[axis] = unpadded - padding;
            if (input_spatial[axis] >= static_cast<std::uint64_t>(
                                           stage.convolution_input_dimensions[
                                               axis + 2U])) {
              in_bounds = false;
              break;
            }
          }
          if (!in_bounds) {
            continue;
          }
          const std::array<std::uint64_t, 5> input_coordinates = {
              output[0],
              input_channel_base + channel,
              input_spatial[0],
              input_spatial[1],
              input_spatial[2]};
          const std::array<std::uint64_t, 5> filter_coordinates = {
              output[1], channel, filter_d, filter_h, filter_w};
          accumulator +=
              static_cast<double>(load_value(
                  input,
                  stage.input_type(0),
                  tensor_offset(input_coordinates,
                                stage.convolution_input_strides))) *
              static_cast<double>(load_value(
                  filter,
                  stage.input_type(1),
                  tensor_offset(filter_coordinates,
                                stage.convolution_filter_strides)));
        }
      }
    }
  }
  if (!std::isfinite(accumulator) ||
      accumulator > static_cast<double>(std::numeric_limits<float>::max()) ||
      accumulator < -static_cast<double>(std::numeric_limits<float>::max())) {
    invalid_oracle("convolution accumulation must be finite");
  }
  return static_cast<float>(accumulator);
}

float absolute_tolerance(const AscendStageArtifact& stage) {
  if (stage.output_type() == StorageDataType::kFloat16) {
    return 1.0e-1F;
  }
  if (stage.output_type() == StorageDataType::kBFloat16) {
    return 2.5e-1F;
  }
  return 5.0e-3F *
         std::sqrt(std::max(
             1.0F,
             static_cast<float>(reduction_element_count(stage)) / 512.0F));
}

float relative_tolerance(const AscendStageArtifact& stage) {
  return stage.output_type() == StorageDataType::kFloat32 ? 5.0e-3F : 7.5e-2F;
}

void validate_one_output(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> input,
    std::span<const std::uint8_t> filter,
    std::span<const std::uint8_t> actual,
    std::uint64_t logical) {
  const auto coordinates = output_coordinates(stage, logical);
  const std::uint64_t offset =
      tensor_offset(coordinates, stage.convolution_output_strides);
  const float expected = expected_value(stage, input, filter, coordinates);
  const float observed = load_value(actual, stage.output_type(), offset);
  const float tolerance = absolute_tolerance(stage) +
                          relative_tolerance(stage) * std::abs(expected);
  if (std::abs(observed - expected) > tolerance) {
    invalid_oracle("convolution output differs from the host oracle");
  }
}

std::vector<bool> logical_output_mask(const AscendStageArtifact& stage) {
  const std::size_t element_count =
      stage.arguments[2].size / storage_element_size(stage.output_type());
  std::vector<bool> written(element_count, false);
  const std::uint64_t logical_count = output_element_count(stage);
  for (std::uint64_t logical = 0U; logical < logical_count; ++logical) {
    const std::uint64_t offset = tensor_offset(
        output_coordinates(stage, logical), stage.convolution_output_strides);
    if (offset >= written.size() || written[offset]) {
      invalid_oracle("convolution output logical offsets overlap");
    }
    written[offset] = true;
  }
  return written;
}

void validate_padding(const AscendStageArtifact& stage,
                      std::span<const std::uint8_t> bytes,
                      const std::vector<bool>& written,
                      const char* message) {
  const std::size_t size = storage_element_size(stage.output_type());
  for (std::size_t element = 0; element < written.size(); ++element) {
    if (!written[element] &&
        !std::all_of(bytes.begin() + element * size,
                     bytes.begin() + (element + 1U) * size,
                     [](std::uint8_t byte) { return byte == kPaddingByte; })) {
      invalid_oracle(message);
    }
  }
}

}  // namespace

std::size_t convolution_fprop_kernel_input_count() noexcept { return 2U; }

std::size_t convolution_fprop_output_argument_index() noexcept { return 2U; }

std::size_t convolution_fprop_tensor_slot_count() noexcept { return 3U; }

std::size_t convolution_fprop_runtime_argument_count() noexcept { return 4U; }

const char* convolution_fprop_output_argument_name() noexcept {
  return "output_ptr";
}

void validate_convolution_fprop_stage_runtime_contract(
    const AscendStageArtifact& stage) {
  if (stage.kernel_family != KernelFamily::kConvolutionFprop ||
      stage.operation != "convolution_fprop" ||
      stage.input_count != convolution_fprop_kernel_input_count() ||
      stage.tensor_storage_data_types.size() !=
          convolution_fprop_tensor_slot_count() ||
      stage.arguments.size() != convolution_fprop_runtime_argument_count() ||
      stage.input_type(0) != stage.input_type(1) ||
      stage.input_type(0) != stage.output_type() ||
      (stage.output_type() != StorageDataType::kFloat32 &&
       stage.output_type() != StorageDataType::kFloat16 &&
       stage.output_type() != StorageDataType::kBFloat16) ||
      stage.convolution_spatial_rank < 1 ||
      stage.convolution_spatial_rank > 3 || stage.convolution_groups <= 0 ||
      stage.convolution_input_channels <= 0 ||
      stage.convolution_output_channels <= 0 ||
      stage.convolution_channels_per_group <= 0 ||
      stage.convolution_input_channels % stage.convolution_groups != 0 ||
      stage.convolution_output_channels % stage.convolution_groups != 0 ||
      stage.convolution_channels_per_group !=
          stage.convolution_input_channels / stage.convolution_groups ||
      stage.convolution_input_dimensions[1] !=
          stage.convolution_input_channels ||
      stage.convolution_filter_dimensions[0] !=
          stage.convolution_output_channels ||
      stage.convolution_filter_dimensions[1] !=
          stage.convolution_channels_per_group ||
      stage.convolution_output_dimensions[1] !=
          stage.convolution_output_channels ||
      stage.convolution_output_dimensions[0] !=
          stage.convolution_input_dimensions[0]) {
    invalid_oracle("convolution stage runtime contract is invalid");
  }
  const std::size_t first_active_spatial =
      5U - static_cast<std::size_t>(stage.convolution_spatial_rank);
  for (std::size_t axis = 2U; axis < first_active_spatial; ++axis) {
    const std::size_t spatial = axis - 2U;
    if (stage.convolution_input_dimensions[axis] != 1 ||
        stage.convolution_filter_dimensions[axis] != 1 ||
        stage.convolution_output_dimensions[axis] != 1 ||
        stage.convolution_input_strides[axis] != 0 ||
        stage.convolution_filter_strides[axis] != 0 ||
        stage.convolution_output_strides[axis] != 0 ||
        stage.convolution_pre_padding[spatial] != 0 ||
        stage.convolution_post_padding[spatial] != 0 ||
        stage.convolution_stride[spatial] != 1 ||
        stage.convolution_dilation[spatial] != 1) {
      invalid_oracle("convolution inactive spatial metadata is invalid");
    }
  }
  for (std::size_t spatial = 0U; spatial < 3U; ++spatial) {
    const std::size_t axis = spatial + 2U;
    if (stage.convolution_input_dimensions[axis] <= 0 ||
        stage.convolution_filter_dimensions[axis] <= 0 ||
        stage.convolution_output_dimensions[axis] <= 0 ||
        stage.convolution_pre_padding[spatial] < 0 ||
        stage.convolution_post_padding[spatial] < 0 ||
        stage.convolution_stride[spatial] <= 0 ||
        stage.convolution_dilation[spatial] <= 0) {
      invalid_oracle("convolution spatial metadata is invalid");
    }
    const std::uint64_t padded = checked_add(
        checked_add(
            static_cast<std::uint64_t>(
                stage.convolution_input_dimensions[axis]),
            static_cast<std::uint64_t>(
                stage.convolution_pre_padding[spatial]),
            "convolution padded input size overflows"),
        static_cast<std::uint64_t>(
            stage.convolution_post_padding[spatial]),
        "convolution padded input size overflows");
    const std::uint64_t effective_filter = checked_add(
        checked_multiply(
            static_cast<std::uint64_t>(
                stage.convolution_filter_dimensions[axis] - 1),
            static_cast<std::uint64_t>(stage.convolution_dilation[spatial]),
            "convolution effective filter size overflows"),
        1U,
        "convolution effective filter size overflows");
    if (padded < effective_filter) {
      invalid_oracle("convolution filter exceeds padded input");
    }
    const std::uint64_t expected_output =
        (padded - effective_filter) /
            static_cast<std::uint64_t>(stage.convolution_stride[spatial]) +
        1U;
    if (expected_output != static_cast<std::uint64_t>(
                               stage.convolution_output_dimensions[axis])) {
      invalid_oracle("convolution output shape is inconsistent");
    }
  }
  validate_non_overlapping_tensor(stage.convolution_input_dimensions,
                                  stage.convolution_input_strides,
                                  "convolution input strides overlap");
  validate_non_overlapping_tensor(stage.convolution_filter_dimensions,
                                  stage.convolution_filter_strides,
                                  "convolution filter strides overlap");
  validate_non_overlapping_tensor(stage.convolution_output_dimensions,
                                  stage.convolution_output_strides,
                                  "convolution output strides overlap");
  const std::uint64_t logical_output = output_element_count(stage);
  if (stage.n_elements <= 0 ||
      logical_output != static_cast<std::uint64_t>(stage.n_elements)) {
    invalid_oracle("convolution output element count is inconsistent");
  }
  validate_pointer_argument(stage.arguments[0], 0U, "input_ptr");
  validate_pointer_argument(stage.arguments[1], 1U, "filter_ptr");
  validate_pointer_argument(stage.arguments[2], 2U, "output_ptr");
  if (stage.arguments[0].uid == stage.arguments[1].uid ||
      stage.arguments[0].uid == stage.arguments[2].uid ||
      stage.arguments[1].uid == stage.arguments[2].uid) {
    invalid_oracle("convolution tensor UIDs must be distinct");
  }
  const std::array<std::size_t, 3> expected_sizes = {
      storage_bytes(stage.convolution_input_dimensions,
                    stage.convolution_input_strides,
                    stage.input_type(0)),
      storage_bytes(stage.convolution_filter_dimensions,
                    stage.convolution_filter_strides,
                    stage.input_type(1)),
      storage_bytes(stage.convolution_output_dimensions,
                    stage.convolution_output_strides,
                    stage.output_type())};
  for (std::size_t index = 0U; index < expected_sizes.size(); ++index) {
    if (stage.arguments[index].size != expected_sizes[index]) {
      invalid_oracle("convolution argument storage span is invalid");
    }
  }
  const ArgumentSource& scalar = stage.arguments[3];
  const auto* scalar_value = std::get_if<std::int32_t>(&scalar.scalar);
  if (scalar.index != 3U || scalar.name != "n_elements" ||
      scalar.source != ArgumentSourceKind::kScalar ||
      scalar.type != RawArgumentType::kI32 || scalar_value == nullptr ||
      *scalar_value != stage.n_elements) {
    invalid_oracle("convolution scalar ABI is invalid");
  }
}

void compute_convolution_fprop_host_oracle(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> input,
    std::span<const std::uint8_t> filter,
    std::span<std::uint8_t> output) {
  validate_convolution_fprop_stage_runtime_contract(stage);
  validate_host_spans(stage, input, filter, output);
  const std::uint64_t logical_count = output_element_count(stage);
  for (std::uint64_t logical = 0U; logical < logical_count; ++logical) {
    const auto coordinates = output_coordinates(stage, logical);
    store_value(output,
                stage.output_type(),
                tensor_offset(coordinates, stage.convolution_output_strides),
                expected_value(stage, input, filter, coordinates));
  }
}

void seed_convolution_fprop_host_inputs(
    const AscendStageArtifact& stage,
    std::span<std::uint8_t> input,
    std::span<std::uint8_t> filter) {
  validate_convolution_fprop_stage_runtime_contract(stage);
  if (input.size() != stage.arguments[0].size ||
      filter.size() != stage.arguments[1].size ||
      spans_overlap(input, filter)) {
    invalid_oracle("convolution seed storage is invalid");
  }
  std::fill(input.begin(), input.end(), kPaddingByte);
  std::fill(filter.begin(), filter.end(), kPaddingByte);
  const auto seed_tensor = [](std::span<std::uint8_t> bytes,
                              StorageDataType type,
                              const std::array<std::int64_t, 5>& dimensions,
                              const std::array<std::int64_t, 5>& strides,
                              std::uint64_t multiplier,
                              std::uint64_t increment,
                              std::uint64_t modulus) {
    std::array<std::uint64_t, 5> coordinate{};
    for (coordinate[0] = 0U;
         coordinate[0] < static_cast<std::uint64_t>(dimensions[0]);
         ++coordinate[0]) {
      for (coordinate[1] = 0U;
           coordinate[1] < static_cast<std::uint64_t>(dimensions[1]);
           ++coordinate[1]) {
        for (coordinate[2] = 0U;
             coordinate[2] < static_cast<std::uint64_t>(dimensions[2]);
             ++coordinate[2]) {
          for (coordinate[3] = 0U;
               coordinate[3] < static_cast<std::uint64_t>(dimensions[3]);
               ++coordinate[3]) {
            for (coordinate[4] = 0U;
                 coordinate[4] < static_cast<std::uint64_t>(dimensions[4]);
                 ++coordinate[4]) {
              const std::uint64_t offset = tensor_offset(coordinate, strides);
              const std::int64_t code = static_cast<std::int64_t>(
                                            (offset * multiplier + increment) %
                                            modulus) -
                                        static_cast<std::int64_t>(modulus / 2U);
              store_value(bytes,
                          type,
                          offset,
                          static_cast<float>(code) * 0.125F);
            }
          }
        }
      }
    }
  };
  seed_tensor(input,
              stage.input_type(0),
              stage.convolution_input_dimensions,
              stage.convolution_input_strides,
              17U,
              5U,
              29U);
  seed_tensor(filter,
              stage.input_type(1),
              stage.convolution_filter_dimensions,
              stage.convolution_filter_strides,
              13U,
              11U,
              31U);
}

void validate_convolution_fprop_host_output(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> input,
    std::span<const std::uint8_t> filter,
    std::span<const std::uint8_t> actual) {
  validate_convolution_fprop_stage_runtime_contract(stage);
  validate_host_spans(stage, input, filter, actual);
  const std::uint64_t logical_count = output_element_count(stage);
  const std::uint64_t samples =
      logical_count <= kFullOracleOutputLimit
          ? logical_count
          : std::min(logical_count, kLargeOracleSampleCount);
  for (std::uint64_t sample = 0U; sample < samples; ++sample) {
    const std::uint64_t logical =
        logical_count <= kFullOracleOutputLimit || samples == 1U
            ? sample
            : sample * (logical_count - 1U) / (samples - 1U);
    validate_one_output(stage, input, filter, actual, logical);
  }
  const std::vector<bool> written = logical_output_mask(stage);
  validate_padding(
      stage, actual, written, "convolution output padding was modified");
}

void validate_convolution_fprop_candidate_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> reference,
    std::span<const std::uint8_t> candidate) {
  validate_convolution_fprop_stage_runtime_contract(stage);
  if (reference.size() != stage.arguments[2].size ||
      candidate.size() != stage.arguments[2].size) {
    invalid_oracle("convolution candidate output storage size is invalid");
  }
  const std::uint64_t logical_count = output_element_count(stage);
  const std::uint64_t samples =
      logical_count <= kFullOracleOutputLimit
          ? logical_count
          : std::min(logical_count, kLargeOracleSampleCount);
  for (std::uint64_t sample = 0U; sample < samples; ++sample) {
    const std::uint64_t logical =
        logical_count <= kFullOracleOutputLimit || samples == 1U
            ? sample
            : sample * (logical_count - 1U) / (samples - 1U);
    const std::uint64_t offset = tensor_offset(
        output_coordinates(stage, logical), stage.convolution_output_strides);
    const float expected = load_value(reference, stage.output_type(), offset);
    const float observed = load_value(candidate, stage.output_type(), offset);
    const float tolerance =
        2.0F * absolute_tolerance(stage) +
        2.0F * relative_tolerance(stage) *
            std::max(std::abs(expected), std::abs(observed));
    if (std::abs(observed - expected) > tolerance) {
      invalid_oracle("convolution autotune candidates differ on output");
    }
  }
  const std::vector<bool> written = logical_output_mask(stage);
  validate_padding(stage,
                   reference,
                   written,
                   "convolution reference output padding was modified");
  const std::size_t size = storage_element_size(stage.output_type());
  for (std::size_t element = 0U; element < written.size(); ++element) {
    if (written[element]) {
      continue;
    }
    const auto reference_begin = reference.begin() + element * size;
    const auto candidate_begin = candidate.begin() + element * size;
    if (!std::equal(reference_begin,
                    reference_begin + size,
                    candidate_begin)) {
      invalid_oracle("convolution autotune candidate modified padding");
    }
  }
}

void commit_convolution_fprop_host_output(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> actual,
    std::span<std::uint8_t> shadow) {
  validate_convolution_fprop_stage_runtime_contract(stage);
  if (actual.size() != stage.arguments[2].size ||
      shadow.size() != stage.arguments[2].size) {
    invalid_oracle("convolution commit storage size is invalid");
  }
  std::copy(actual.begin(), actual.end(), shadow.begin());
}

}  // namespace flagdnn::ascend
