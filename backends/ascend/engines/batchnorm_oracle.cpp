/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/batchnorm_oracle.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace flagdnn::ascend {
namespace {

constexpr std::size_t kMaximumRank = 8U;
constexpr std::size_t kWorkspaceAlignment = 256U;
constexpr std::uint8_t kPaddingSentinel = 0xA5U;

[[noreturn]] void invalid(const char* message) {
  throw std::invalid_argument(message);
}

std::size_t element_size(StorageDataType type) {
  switch (type) {
    case StorageDataType::kFloat32:
      return 4U;
    case StorageDataType::kFloat16:
    case StorageDataType::kBFloat16:
      return 2U;
    default:
      invalid("BatchNorm storage data type is invalid");
  }
}

float half_to_float(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U)
                             << 16U;
  const std::uint32_t exponent = (value >> 10U) & 0x1FU;
  std::uint32_t fraction = value & 0x3FFU;
  std::uint32_t bits = 0U;
  if (exponent == 0U) {
    if (fraction == 0U) {
      bits = sign;
    } else {
      std::uint32_t normalized = 113U;
      while ((fraction & 0x400U) == 0U) {
        fraction <<= 1U;
        --normalized;
      }
      bits = sign | (normalized << 23U) |
             ((fraction & 0x3FFU) << 13U);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (fraction << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
  }
  return std::bit_cast<float>(bits);
}

std::uint16_t float_to_half(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
  const std::uint32_t fraction = bits & 0x7FFFFFU;
  if (exponent == 0xFFU) {
    return static_cast<std::uint16_t>(
        sign | 0x7C00U | (fraction == 0U ? 0U : 0x200U));
  }
  const std::int32_t encoded_exponent =
      static_cast<std::int32_t>(exponent) - 112;
  if (encoded_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (encoded_exponent <= 0) {
    if (encoded_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t mantissa = fraction | 0x800000U;
    const std::uint32_t shift =
        static_cast<std::uint32_t>(14 - encoded_exponent);
    const std::uint32_t rounded = mantissa + ((1U << (shift - 1U)) - 1U) +
                                  ((mantissa >> shift) & 1U);
    return static_cast<std::uint16_t>(sign | (rounded >> shift));
  }
  const std::uint32_t rounded =
      fraction + 0xFFFU + ((fraction >> 13U) & 1U);
  std::uint32_t result =
      sign | (static_cast<std::uint32_t>(encoded_exponent) << 10U) |
      (rounded >> 13U);
  if ((rounded & 0x800000U) != 0U) {
    result = sign |
             (static_cast<std::uint32_t>(encoded_exponent + 1) << 10U);
  }
  return static_cast<std::uint16_t>(result);
}

float load(std::span<const std::uint8_t> bytes,
           StorageDataType type,
           std::size_t index) {
  const std::size_t size = element_size(type);
  if (bytes.size() % size != 0U || index >= bytes.size() / size) {
    invalid("BatchNorm host storage is too small");
  }
  float result = 0.0F;
  if (type == StorageDataType::kFloat32) {
    std::memcpy(&result, bytes.data() + index * size, sizeof(result));
  } else {
    std::uint16_t encoded = 0U;
    std::memcpy(&encoded, bytes.data() + index * size, sizeof(encoded));
    result = type == StorageDataType::kFloat16
                 ? half_to_float(encoded)
                 : std::bit_cast<float>(static_cast<std::uint32_t>(encoded)
                                        << 16U);
  }
  if (!std::isfinite(result)) {
    invalid("BatchNorm host value must be finite");
  }
  return result;
}

void store(std::span<std::uint8_t> bytes,
           StorageDataType type,
           std::size_t index,
           float value) {
  if (!std::isfinite(value)) {
    invalid("BatchNorm host result must be finite");
  }
  const std::size_t size = element_size(type);
  if (bytes.size() % size != 0U || index >= bytes.size() / size) {
    invalid("BatchNorm host storage is too small");
  }
  if (type == StorageDataType::kFloat32) {
    std::memcpy(bytes.data() + index * size, &value, sizeof(value));
    return;
  }
  std::uint16_t encoded = 0U;
  if (type == StorageDataType::kFloat16) {
    encoded = float_to_half(value);
    if ((encoded & 0x7C00U) == 0x7C00U) {
      invalid("BatchNorm float16 result must be finite");
    }
  } else {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    encoded = static_cast<std::uint16_t>(bits >> 16U);
    if ((encoded & 0x7F80U) == 0x7F80U) {
      invalid("BatchNorm bfloat16 result must be finite");
    }
  }
  std::memcpy(bytes.data() + index * size, &encoded, sizeof(encoded));
}

std::size_t tensor_offset(
    std::size_t logical,
    const std::array<std::int64_t, kMaximumRank>& dimensions,
    const std::array<std::int64_t, kMaximumRank>& strides) {
  std::size_t offset = 0U;
  for (std::size_t reversed = kMaximumRank; reversed != 0U; --reversed) {
    const std::size_t axis = reversed - 1U;
    if (dimensions[axis] <= 0 || strides[axis] < 0) {
      invalid("BatchNorm tensor metadata is invalid");
    }
    const std::size_t dimension =
        static_cast<std::size_t>(dimensions[axis]);
    const std::size_t coordinate = logical % dimension;
    logical /= dimension;
    const std::size_t stride = static_cast<std::size_t>(strides[axis]);
    if (coordinate != 0U &&
        stride > std::numeric_limits<std::size_t>::max() / coordinate) {
      invalid("BatchNorm tensor offset overflows");
    }
    const std::size_t term = coordinate * stride;
    if (offset > std::numeric_limits<std::size_t>::max() - term) {
      invalid("BatchNorm tensor offset overflows");
    }
    offset += term;
  }
  if (logical != 0U) {
    invalid("BatchNorm logical index exceeds tensor shape");
  }
  return offset;
}

void validate_nonoverlap(
    const std::array<std::int64_t, kMaximumRank>& dimensions,
    const std::array<std::int64_t, kMaximumRank>& strides,
    std::size_t first_axis) {
  std::vector<std::pair<std::size_t, std::size_t>> active;
  for (std::size_t axis = first_axis; axis < kMaximumRank; ++axis) {
    if (dimensions[axis] > 1) {
      if (strides[axis] <= 0) {
        invalid("BatchNorm tensor strides alias logical elements");
      }
      active.emplace_back(static_cast<std::size_t>(strides[axis]),
                          static_cast<std::size_t>(dimensions[axis]));
    }
  }
  std::sort(active.begin(), active.end());
  std::size_t span = 1U;
  for (const auto& [stride, dimension] : active) {
    if (stride < span ||
        dimension - 1U >
            (std::numeric_limits<std::size_t>::max() - span) / stride) {
      invalid("BatchNorm tensor strides alias logical elements");
    }
    span += (dimension - 1U) * stride;
  }
}

std::size_t required_bytes(
    const std::array<std::int64_t, kMaximumRank>& dimensions,
    const std::array<std::int64_t, kMaximumRank>& strides,
    StorageDataType type) {
  std::size_t maximum = 0U;
  for (std::size_t axis = 0; axis < kMaximumRank; ++axis) {
    const std::size_t coordinate =
        static_cast<std::size_t>(dimensions[axis] - 1);
    const std::size_t stride = static_cast<std::size_t>(strides[axis]);
    if (coordinate != 0U &&
        stride > std::numeric_limits<std::size_t>::max() / coordinate) {
      invalid("BatchNorm tensor storage size overflows");
    }
    const std::size_t term = coordinate * stride;
    if (maximum > std::numeric_limits<std::size_t>::max() - term) {
      invalid("BatchNorm tensor storage size overflows");
    }
    maximum += term;
  }
  if (maximum == std::numeric_limits<std::size_t>::max() ||
      maximum + 1U >
          std::numeric_limits<std::size_t>::max() / element_size(type)) {
    invalid("BatchNorm tensor storage size overflows");
  }
  return (maximum + 1U) * element_size(type);
}

bool overlaps(std::span<const std::uint8_t> left,
              std::span<const std::uint8_t> right) {
  if (left.empty() || right.empty()) {
    return false;
  }
  const auto left_start = reinterpret_cast<std::uintptr_t>(left.data());
  const auto right_start = reinterpret_cast<std::uintptr_t>(right.data());
  if (left_start > std::numeric_limits<std::uintptr_t>::max() - left.size() ||
      right_start >
          std::numeric_limits<std::uintptr_t>::max() - right.size()) {
    invalid("BatchNorm host address range overflows");
  }
  return left_start < right_start + right.size() &&
         right_start < left_start + left.size();
}

void validate_disjoint(ConstBatchNormHostBuffers inputs,
                       ConstBatchNormHostBuffers outputs) {
  std::array<std::span<const std::uint8_t>, 10> all = {
      inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
      outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]};
  for (std::size_t left = 0U; left < all.size(); ++left) {
    for (std::size_t right = left + 1U; right < all.size(); ++right) {
      if (overlaps(all[left], all[right])) {
        invalid("BatchNorm host tensor storage overlaps");
      }
    }
  }
}

void validate_sizes(const AscendStageArtifact& stage,
                    ConstBatchNormHostBuffers inputs,
                    ConstBatchNormHostBuffers outputs) {
  for (std::size_t index = 0U; index < inputs.size(); ++index) {
    if (inputs[index].size() != stage.arguments[index].size) {
      invalid("BatchNorm host input size is invalid");
    }
  }
  for (std::size_t index = 0U; index < outputs.size(); ++index) {
    if (outputs[index].size() != stage.arguments[index + 5U].size) {
      invalid("BatchNorm host output size is invalid");
    }
  }
}

void require_close(float expected,
                   float actual,
                   StorageDataType type,
                   const char* message) {
  const float absolute = type == StorageDataType::kFloat32
                             ? 3.0e-5F
                             : type == StorageDataType::kFloat16 ? 3.0e-3F
                                                                : 3.0e-2F;
  const float relative = type == StorageDataType::kFloat32
                             ? 4.0e-4F
                             : type == StorageDataType::kFloat16 ? 3.0e-2F
                                                                : 8.0e-2F;
  if (!std::isfinite(expected) || !std::isfinite(actual) ||
      std::abs(actual - expected) >
          absolute + relative * std::abs(expected)) {
    invalid(message);
  }
}

void validate_padding(const AscendStageArtifact& stage,
                      std::span<const std::uint8_t> y) {
  const std::size_t size = element_size(stage.output_type());
  std::vector<bool> logical(y.size() / size, false);
  for (std::int32_t index = 0; index < stage.n_elements; ++index) {
    logical.at(tensor_offset(static_cast<std::size_t>(index),
                             stage.batchnorm_dimensions,
                             stage.batchnorm_y_strides)) = true;
  }
  for (std::size_t element = 0U; element < logical.size(); ++element) {
    if (!logical[element]) {
      for (std::size_t byte = 0U; byte < size; ++byte) {
        if (y[element * size + byte] != kPaddingSentinel) {
          invalid("BatchNorm candidate modified Y padding");
        }
      }
    }
  }
}

}  // namespace

std::size_t batchnorm_kernel_input_count() noexcept { return 5U; }
std::size_t batchnorm_tensor_slot_count() noexcept { return 10U; }
std::size_t batchnorm_runtime_argument_count() noexcept { return 11U; }
std::size_t batchnorm_first_output_argument_index() noexcept { return 5U; }

void validate_batchnorm_stage_runtime_contract(const AscendStageArtifact& stage) {
  if (stage.operation != "batchnorm" ||
      stage.kernel_family != KernelFamily::kBatchNorm ||
      stage.input_count != batchnorm_kernel_input_count() ||
      stage.tensor_storage_data_types.size() != batchnorm_tensor_slot_count() ||
      stage.arguments.size() != batchnorm_runtime_argument_count() ||
      stage.n_elements <= 0 || stage.batchnorm_rank < 2 ||
      stage.batchnorm_rank > 8 || stage.batchnorm_batch <= 0 ||
      stage.batchnorm_channels <= 0 || stage.batchnorm_spatial <= 0 ||
      stage.batchnorm_reduction_elements <= 0 ||
      !std::isfinite(stage.batchnorm_epsilon) ||
      stage.batchnorm_epsilon <= 0.0 ||
      !std::isfinite(stage.batchnorm_momentum) ||
      stage.batchnorm_momentum < 0.0 || stage.batchnorm_momentum > 1.0) {
    invalid("BatchNorm stage runtime contract is invalid");
  }
  const StorageDataType type = stage.input_type(0U);
  if ((type != StorageDataType::kFloat32 &&
       type != StorageDataType::kFloat16 &&
       type != StorageDataType::kBFloat16) ||
      stage.input_type(1U) != type || stage.input_type(2U) != type ||
      stage.tensor_storage_data_types[5U] != type) {
    invalid("BatchNorm X/scale/bias/Y data types are invalid");
  }
  for (std::size_t index : {3U, 4U, 6U, 7U, 8U, 9U}) {
    if (stage.tensor_storage_data_types[index] != StorageDataType::kFloat32) {
      invalid("BatchNorm statistic data types are invalid");
    }
  }
  const std::size_t first_axis =
      kMaximumRank - static_cast<std::size_t>(stage.batchnorm_rank);
  std::uint64_t elements = 1U;
  std::uint64_t spatial = 1U;
  for (std::size_t axis = 0U; axis < kMaximumRank; ++axis) {
    const std::int64_t dimension = stage.batchnorm_dimensions[axis];
    if (dimension <= 0 || stage.batchnorm_x_strides[axis] < 0 ||
        stage.batchnorm_y_strides[axis] < 0 ||
        (axis < first_axis &&
         (dimension != 1 || stage.batchnorm_x_strides[axis] != 0 ||
          stage.batchnorm_y_strides[axis] != 0))) {
      invalid("BatchNorm tensor metadata is invalid");
    }
    const auto encoded = static_cast<std::uint64_t>(dimension);
    if (elements > std::numeric_limits<std::uint64_t>::max() / encoded) {
      invalid("BatchNorm element count overflows");
    }
    elements *= encoded;
    if (axis > first_axis + 1U) {
      if (spatial > std::numeric_limits<std::uint64_t>::max() / encoded) {
        invalid("BatchNorm spatial count overflows");
      }
      spatial *= encoded;
    }
  }
  if (elements != static_cast<std::uint64_t>(stage.n_elements) ||
      stage.batchnorm_dimensions[first_axis] != stage.batchnorm_batch ||
      stage.batchnorm_dimensions[first_axis + 1U] !=
          stage.batchnorm_channels ||
      spatial != static_cast<std::uint64_t>(stage.batchnorm_spatial) ||
      static_cast<std::int64_t>(stage.batchnorm_batch *
                                stage.batchnorm_spatial) !=
          stage.batchnorm_reduction_elements) {
    invalid("BatchNorm decomposition differs from tensor shape");
  }
  validate_nonoverlap(stage.batchnorm_dimensions,
                      stage.batchnorm_x_strides,
                      first_axis);
  validate_nonoverlap(stage.batchnorm_dimensions,
                      stage.batchnorm_y_strides,
                      first_axis);
  static constexpr std::array<const char*, 10> names = {
      "x_ptr", "scale_ptr", "bias_ptr", "previous_running_mean_ptr",
      "previous_running_variance_ptr", "y_ptr", "mean_ptr",
      "inv_variance_ptr", "next_running_mean_ptr",
      "next_running_variance_ptr"};
  const std::size_t x_bytes = required_bytes(stage.batchnorm_dimensions,
                                             stage.batchnorm_x_strides,
                                             type);
  const std::size_t y_bytes = required_bytes(stage.batchnorm_dimensions,
                                             stage.batchnorm_y_strides,
                                             type);
  const std::size_t value_bytes =
      static_cast<std::size_t>(stage.batchnorm_channels) * element_size(type);
  const std::size_t statistic_bytes =
      static_cast<std::size_t>(stage.batchnorm_channels) * sizeof(float);
  const std::array<std::size_t, 10> sizes = {
      x_bytes, value_bytes, value_bytes, statistic_bytes, statistic_bytes,
      y_bytes, statistic_bytes, statistic_bytes, statistic_bytes,
      statistic_bytes};
  std::set<std::int64_t> uids;
  for (std::size_t index = 0U; index < names.size(); ++index) {
    const ArgumentSource& argument = stage.arguments[index];
    if (argument.index != index || argument.name != names[index] ||
        argument.type != RawArgumentType::kPointer ||
        (argument.source != ArgumentSourceKind::kBinding &&
         argument.source != ArgumentSourceKind::kGraphWorkspace) ||
        argument.uid <= 0 || !uids.insert(argument.uid).second ||
        argument.size != sizes[index] || argument.alignment == 0U ||
        (argument.alignment & (argument.alignment - 1U)) != 0U ||
        (argument.source == ArgumentSourceKind::kBinding &&
         argument.workspace_offset != 0U) ||
        (argument.source == ArgumentSourceKind::kGraphWorkspace &&
         (argument.alignment != kWorkspaceAlignment ||
          argument.workspace_offset % kWorkspaceAlignment != 0U))) {
      invalid("BatchNorm runtime pointer ABI is invalid");
    }
  }
  const ArgumentSource& scalar = stage.arguments[10U];
  const auto* value = std::get_if<std::int32_t>(&scalar.scalar);
  if (scalar.index != 10U || scalar.name != "n_elements" ||
      scalar.source != ArgumentSourceKind::kScalar ||
      scalar.type != RawArgumentType::kI32 || value == nullptr ||
      *value != stage.n_elements) {
    invalid("BatchNorm runtime scalar ABI is invalid");
  }
}

void seed_batchnorm_host_inputs(const AscendStageArtifact& stage,
                                BatchNormHostBuffers inputs) {
  validate_batchnorm_stage_runtime_contract(stage);
  ConstBatchNormHostBuffers const_inputs = {
      inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]};
  std::array<std::vector<std::uint8_t>, 5> probes = {
      std::vector<std::uint8_t>(stage.arguments[5].size, kPaddingSentinel),
      std::vector<std::uint8_t>(stage.arguments[6].size, kPaddingSentinel),
      std::vector<std::uint8_t>(stage.arguments[7].size, kPaddingSentinel),
      std::vector<std::uint8_t>(stage.arguments[8].size, kPaddingSentinel),
      std::vector<std::uint8_t>(stage.arguments[9].size, kPaddingSentinel)};
  ConstBatchNormHostBuffers const_probes = {
      probes[0], probes[1], probes[2], probes[3], probes[4]};
  validate_sizes(stage, const_inputs, const_probes);
  const StorageDataType type = stage.input_type(0U);
  for (std::int64_t channel = 0; channel < stage.batchnorm_channels;
       ++channel) {
    const float scale = channel % 2 == 0 ? 1.5F : -0.75F;
    const float bias = static_cast<float>(channel - 1) * 0.25F;
    const float previous_mean = static_cast<float>(channel) - 0.5F;
    const float previous_variance = 1.0F + 0.5F * channel;
    store(inputs[1], type, static_cast<std::size_t>(channel), scale);
    store(inputs[2], type, static_cast<std::size_t>(channel), bias);
    store(inputs[3], StorageDataType::kFloat32,
          static_cast<std::size_t>(channel), previous_mean);
    store(inputs[4], StorageDataType::kFloat32,
          static_cast<std::size_t>(channel), previous_variance);
  }
  const std::size_t channels =
      static_cast<std::size_t>(stage.batchnorm_channels);
  const std::size_t spatial =
      static_cast<std::size_t>(stage.batchnorm_spatial);
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const std::size_t index = static_cast<std::size_t>(logical);
    const std::size_t channel = (index / spatial) % channels;
    const float baseline = channel % 2U == 0U ? 32.0F : -32.0F;
    const float delta =
        static_cast<float>(static_cast<int>(index % 7U) - 3) *
        (type == StorageDataType::kFloat32
             ? 0.125F
             : type == StorageDataType::kFloat16 ? 0.25F : 0.5F);
    store(inputs[0],
          type,
          tensor_offset(index, stage.batchnorm_dimensions,
                        stage.batchnorm_x_strides),
          baseline + delta);
  }
}

void compute_batchnorm_host_oracle(const AscendStageArtifact& stage,
                                   ConstBatchNormHostBuffers inputs,
                                   BatchNormHostBuffers outputs) {
  validate_batchnorm_stage_runtime_contract(stage);
  ConstBatchNormHostBuffers const_outputs = {
      outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]};
  validate_sizes(stage, inputs, const_outputs);
  validate_disjoint(inputs, const_outputs);
  const StorageDataType type = stage.input_type(0U);
  const std::size_t channels =
      static_cast<std::size_t>(stage.batchnorm_channels);
  const std::size_t spatial =
      static_cast<std::size_t>(stage.batchnorm_spatial);
  const std::size_t reduction =
      static_cast<std::size_t>(stage.batchnorm_reduction_elements);
  std::vector<double> sums(channels, 0.0);
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const std::size_t index = static_cast<std::size_t>(logical);
    const std::size_t channel = (index / spatial) % channels;
    sums[channel] += load(inputs[0], type,
                          tensor_offset(index, stage.batchnorm_dimensions,
                                        stage.batchnorm_x_strides));
  }
  std::vector<double> means(channels, 0.0);
  for (std::size_t channel = 0U; channel < channels; ++channel) {
    means[channel] = sums[channel] / static_cast<double>(reduction);
  }
  std::vector<double> squares(channels, 0.0);
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const std::size_t index = static_cast<std::size_t>(logical);
    const std::size_t channel = (index / spatial) % channels;
    const double centered =
        load(inputs[0], type,
             tensor_offset(index, stage.batchnorm_dimensions,
                           stage.batchnorm_x_strides)) -
        means[channel];
    squares[channel] += centered * centered;
  }
  std::vector<double> invs(channels, 0.0);
  for (std::size_t channel = 0U; channel < channels; ++channel) {
    const double variance = squares[channel] / static_cast<double>(reduction);
    invs[channel] = 1.0 / std::sqrt(variance + stage.batchnorm_epsilon);
    const double unbiased =
        reduction > 1U
            ? variance * static_cast<double>(reduction) /
                  static_cast<double>(reduction - 1U)
            : variance;
    store(outputs[1], StorageDataType::kFloat32, channel,
          static_cast<float>(means[channel]));
    store(outputs[2], StorageDataType::kFloat32, channel,
          static_cast<float>(invs[channel]));
    store(outputs[3], StorageDataType::kFloat32, channel,
          static_cast<float>(
              load(inputs[3], StorageDataType::kFloat32, channel) *
                  (1.0 - stage.batchnorm_momentum) +
              means[channel] * stage.batchnorm_momentum));
    store(outputs[4], StorageDataType::kFloat32, channel,
          static_cast<float>(
              load(inputs[4], StorageDataType::kFloat32, channel) *
                  (1.0 - stage.batchnorm_momentum) +
              unbiased * stage.batchnorm_momentum));
  }
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const std::size_t index = static_cast<std::size_t>(logical);
    const std::size_t channel = (index / spatial) % channels;
    const float x = load(inputs[0], type,
                         tensor_offset(index, stage.batchnorm_dimensions,
                                       stage.batchnorm_x_strides));
    const float scale = load(inputs[1], type, channel);
    const float bias = load(inputs[2], type, channel);
    store(outputs[0], type,
          tensor_offset(index, stage.batchnorm_dimensions,
                        stage.batchnorm_y_strides),
          static_cast<float>((x - means[channel]) * invs[channel] * scale +
                             bias));
  }
}

void validate_batchnorm_host_outputs(const AscendStageArtifact& stage,
                                     ConstBatchNormHostBuffers inputs,
                                     ConstBatchNormHostBuffers outputs) {
  validate_batchnorm_stage_runtime_contract(stage);
  validate_sizes(stage, inputs, outputs);
  validate_disjoint(inputs, outputs);
  std::array<std::vector<std::uint8_t>, 5> expected_storage = {
      std::vector<std::uint8_t>(outputs[0].size(), kPaddingSentinel),
      std::vector<std::uint8_t>(outputs[1].size(), kPaddingSentinel),
      std::vector<std::uint8_t>(outputs[2].size(), kPaddingSentinel),
      std::vector<std::uint8_t>(outputs[3].size(), kPaddingSentinel),
      std::vector<std::uint8_t>(outputs[4].size(), kPaddingSentinel)};
  BatchNormHostBuffers expected = {
      expected_storage[0], expected_storage[1], expected_storage[2],
      expected_storage[3], expected_storage[4]};
  compute_batchnorm_host_oracle(stage, inputs, expected);
  const StorageDataType type = stage.output_type();
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const std::size_t offset = tensor_offset(
        static_cast<std::size_t>(logical), stage.batchnorm_dimensions,
        stage.batchnorm_y_strides);
    require_close(load(expected_storage[0], type, offset),
                  load(outputs[0], type, offset), type,
                  "BatchNorm Y differs from host oracle");
  }
  validate_padding(stage, outputs[0]);
  for (std::size_t output = 1U; output < outputs.size(); ++output) {
    for (std::int64_t channel = 0; channel < stage.batchnorm_channels;
         ++channel) {
      require_close(load(expected_storage[output], StorageDataType::kFloat32,
                         static_cast<std::size_t>(channel)),
                    load(outputs[output], StorageDataType::kFloat32,
                         static_cast<std::size_t>(channel)),
                    StorageDataType::kFloat32,
                    "BatchNorm statistic differs from host oracle");
    }
  }
}

void validate_batchnorm_candidate_outputs(
    const AscendStageArtifact& stage,
    ConstBatchNormHostBuffers reference,
    ConstBatchNormHostBuffers candidate) {
  validate_batchnorm_stage_runtime_contract(stage);
  const StorageDataType type = stage.output_type();
  for (std::size_t output = 0U; output < reference.size(); ++output) {
    if (reference[output].size() != stage.arguments[output + 5U].size ||
        candidate[output].size() != reference[output].size()) {
      invalid("BatchNorm candidate output size is invalid");
    }
  }
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const std::size_t offset = tensor_offset(
        static_cast<std::size_t>(logical), stage.batchnorm_dimensions,
        stage.batchnorm_y_strides);
    require_close(load(reference[0], type, offset),
                  load(candidate[0], type, offset), type,
                  "BatchNorm candidates disagree on Y");
  }
  validate_padding(stage, reference[0]);
  validate_padding(stage, candidate[0]);
  for (std::size_t output = 1U; output < reference.size(); ++output) {
    for (std::int64_t channel = 0; channel < stage.batchnorm_channels;
         ++channel) {
      require_close(load(reference[output], StorageDataType::kFloat32,
                         static_cast<std::size_t>(channel)),
                    load(candidate[output], StorageDataType::kFloat32,
                         static_cast<std::size_t>(channel)),
                    StorageDataType::kFloat32,
                    "BatchNorm candidates disagree on a statistic");
    }
  }
}

void commit_batchnorm_host_outputs(const AscendStageArtifact& stage,
                                   ConstBatchNormHostBuffers actual,
                                   BatchNormHostBuffers shadow) {
  validate_batchnorm_stage_runtime_contract(stage);
  for (std::size_t output = 0U; output < actual.size(); ++output) {
    if (actual[output].size() != stage.arguments[output + 5U].size ||
        shadow[output].size() != actual[output].size()) {
      invalid("BatchNorm commit output size is invalid");
    }
    std::copy(actual[output].begin(), actual[output].end(),
              shadow[output].begin());
  }
}

}  // namespace flagdnn::ascend
