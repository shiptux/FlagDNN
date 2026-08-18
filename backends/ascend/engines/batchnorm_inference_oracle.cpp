/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/batchnorm_inference_oracle.hpp"

#include <algorithm>
#include <array>
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

constexpr std::size_t kMaximumRank = 8;
constexpr std::size_t kGraphWorkspaceAlignment = 256;
constexpr std::uint8_t kPaddingSentinel = 0xA5U;

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
    default:
      invalid_oracle("batchnorm oracle storage data type is invalid");
  }
}

float half_to_float(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  const std::uint32_t exponent = (value >> 10U) & 0x1FU;
  std::uint32_t fraction = value & 0x3FFU;
  std::uint32_t bits = 0;
  if (exponent == 0U) {
    if (fraction == 0U) {
      bits = sign;
    } else {
      std::uint32_t normalized_exponent = 113U;
      while ((fraction & 0x400U) == 0U) {
        fraction <<= 1U;
        --normalized_exponent;
      }
      bits = sign | (normalized_exponent << 23U) |
             ((fraction & 0x3FFU) << 13U);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (fraction << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
  }
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::uint16_t float_to_half(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
  const std::uint32_t fraction = bits & 0x7FFFFFU;
  if (exponent == 0xFFU) {
    return static_cast<std::uint16_t>(
        sign | 0x7C00U | (fraction == 0U ? 0U : 0x200U));
  }
  const std::int32_t half_exponent = static_cast<std::int32_t>(exponent) - 112;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t mantissa = fraction | 0x800000U;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - half_exponent);
    const std::uint32_t rounded = mantissa + ((1U << (shift - 1U)) - 1U) +
                                  ((mantissa >> shift) & 1U);
    return static_cast<std::uint16_t>(sign | (rounded >> shift));
  }
  const std::uint32_t rounded =
      fraction + 0xFFFU + ((fraction >> 13U) & 1U);
  std::uint32_t result =
      sign | (static_cast<std::uint32_t>(half_exponent) << 10U) |
      (rounded >> 13U);
  if ((rounded & 0x800000U) != 0U) {
    result = sign | (static_cast<std::uint32_t>(half_exponent + 1) << 10U);
  }
  return static_cast<std::uint16_t>(result);
}

float load_value(std::span<const std::uint8_t> storage,
                 StorageDataType type,
                 std::uint64_t element,
                 const char* range_error) {
  const std::size_t size = storage_element_size(type);
  if (storage.size() % size != 0U || element >= storage.size() / size) {
    invalid_oracle(range_error);
  }
  float result = 0.0F;
  if (type == StorageDataType::kFloat32) {
    std::memcpy(&result, storage.data() + element * size, sizeof(result));
  } else {
    std::uint16_t encoded = 0;
    std::memcpy(&encoded, storage.data() + element * size, sizeof(encoded));
    if (type == StorageDataType::kFloat16) {
      result = half_to_float(encoded);
    } else {
      const std::uint32_t bits = static_cast<std::uint32_t>(encoded) << 16U;
      std::memcpy(&result, &bits, sizeof(result));
    }
  }
  if (!std::isfinite(result)) {
    invalid_oracle("batchnorm oracle inputs and parameters must be finite");
  }
  return result;
}

void store_value(std::span<std::uint8_t> storage,
                 StorageDataType type,
                 std::uint64_t element,
                 float value,
                 const char* range_error) {
  if (!std::isfinite(value)) {
    invalid_oracle("batchnorm oracle result must be finite");
  }
  const std::size_t size = storage_element_size(type);
  if (storage.size() % size != 0U || element >= storage.size() / size) {
    invalid_oracle(range_error);
  }
  if (type == StorageDataType::kFloat32) {
    std::memcpy(storage.data() + element * size, &value, sizeof(value));
    return;
  }
  std::uint16_t encoded = 0;
  if (type == StorageDataType::kFloat16) {
    encoded = float_to_half(value);
    if ((encoded & 0x7C00U) == 0x7C00U) {
      invalid_oracle("batchnorm float16 result must be finite");
    }
  } else {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    encoded = static_cast<std::uint16_t>(bits >> 16U);
    if ((encoded & 0x7F80U) == 0x7F80U) {
      invalid_oracle("batchnorm bfloat16 result must be finite");
    }
  }
  std::memcpy(storage.data() + element * size, &encoded, sizeof(encoded));
}

std::uint64_t tensor_offset(
    std::uint64_t logical,
    const std::array<std::int64_t, kMaximumRank>& dimensions,
    const std::array<std::int64_t, kMaximumRank>& strides) {
  std::uint64_t result = 0;
  for (std::size_t reversed = kMaximumRank; reversed != 0; --reversed) {
    const std::size_t axis = reversed - 1U;
    if (dimensions[axis] <= 0 || strides[axis] < 0) {
      invalid_oracle("batchnorm oracle tensor metadata is invalid");
    }
    const auto dimension = static_cast<std::uint64_t>(dimensions[axis]);
    const std::uint64_t coordinate = logical % dimension;
    logical /= dimension;
    const auto stride = static_cast<std::uint64_t>(strides[axis]);
    if (coordinate != 0U &&
        stride > std::numeric_limits<std::uint64_t>::max() / coordinate) {
      invalid_oracle("batchnorm oracle tensor offset overflows");
    }
    const std::uint64_t term = coordinate * stride;
    if (result > std::numeric_limits<std::uint64_t>::max() - term) {
      invalid_oracle("batchnorm oracle tensor offset overflows");
    }
    result += term;
  }
  if (logical != 0U) {
    invalid_oracle("batchnorm oracle logical index exceeds tensor shape");
  }
  return result;
}

void validate_non_overlapping_tensor(
    const std::array<std::int64_t, kMaximumRank>& dimensions,
    const std::array<std::int64_t, kMaximumRank>& strides,
    std::size_t first_axis,
    const char* message) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> active;
  for (std::size_t axis = first_axis; axis < kMaximumRank; ++axis) {
    if (dimensions[axis] > 1) {
      if (strides[axis] <= 0) {
        invalid_oracle(message);
      }
      active.emplace_back(static_cast<std::uint64_t>(strides[axis]),
                          static_cast<std::uint64_t>(dimensions[axis]));
    }
  }
  std::sort(active.begin(), active.end());
  std::uint64_t span = 1;
  for (const auto& [stride, dimension] : active) {
    if (stride < span ||
        dimension - 1U >
            (std::numeric_limits<std::uint64_t>::max() - span) / stride) {
      invalid_oracle(message);
    }
    span += (dimension - 1U) * stride;
  }
}

bool spans_overlap(std::span<const std::uint8_t> left,
                   std::span<const std::uint8_t> right) {
  if (left.empty() || right.empty()) {
    return false;
  }
  const auto left_start =
      reinterpret_cast<std::uintptr_t>(left.data());
  const auto right_start =
      reinterpret_cast<std::uintptr_t>(right.data());
  if (left_start > std::numeric_limits<std::uintptr_t>::max() - left.size() ||
      right_start >
          std::numeric_limits<std::uintptr_t>::max() - right.size()) {
    invalid_oracle("batchnorm host buffer address range overflows");
  }
  const std::uintptr_t left_end = left_start + left.size();
  const std::uintptr_t right_end = right_start + right.size();
  return left_start < right_end && right_start < left_end;
}

void validate_output_disjoint(
    std::span<const std::uint8_t> x,
    std::span<const std::uint8_t> mean,
    std::span<const std::uint8_t> inv_variance,
    std::span<const std::uint8_t> scale,
    std::span<const std::uint8_t> bias,
    std::span<const std::uint8_t> output) {
  for (const std::span<const std::uint8_t> input :
       {x, mean, inv_variance, scale, bias}) {
    if (spans_overlap(input, output)) {
      invalid_oracle("batchnorm output storage overlaps an input");
    }
  }
}

void validate_pointer_argument(const ArgumentSource& argument,
                               std::size_t index,
                               const char* name,
                               bool parameter) {
  if (argument.index != index || argument.name != name ||
      argument.type != RawArgumentType::kPointer ||
      (argument.source != ArgumentSourceKind::kBinding &&
       argument.source != ArgumentSourceKind::kGraphWorkspace) ||
      (parameter && argument.source != ArgumentSourceKind::kBinding)) {
    invalid_oracle("batchnorm stage runtime pointer ABI is invalid");
  }
}

bool valid_alignment(std::size_t alignment) {
  return alignment != 0U && (alignment & (alignment - 1U)) == 0U;
}

std::size_t required_tensor_bytes(
    const std::array<std::int64_t, kMaximumRank>& dimensions,
    const std::array<std::int64_t, kMaximumRank>& strides,
    StorageDataType type) {
  std::uint64_t maximum_offset = 0;
  for (std::size_t axis = 0; axis < kMaximumRank; ++axis) {
    if (dimensions[axis] <= 0 || strides[axis] < 0) {
      invalid_oracle("batchnorm tensor storage metadata is invalid");
    }
    const auto coordinate =
        static_cast<std::uint64_t>(dimensions[axis] - 1);
    const auto stride = static_cast<std::uint64_t>(strides[axis]);
    if (coordinate != 0U &&
        stride > std::numeric_limits<std::uint64_t>::max() / coordinate) {
      invalid_oracle("batchnorm tensor storage size overflows");
    }
    const std::uint64_t term = coordinate * stride;
    if (maximum_offset > std::numeric_limits<std::uint64_t>::max() - term) {
      invalid_oracle("batchnorm tensor storage size overflows");
    }
    maximum_offset += term;
  }
  const std::size_t element_size = storage_element_size(type);
  if (maximum_offset >=
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                 element_size)) {
    invalid_oracle("batchnorm tensor storage size overflows");
  }
  return (static_cast<std::size_t>(maximum_offset) + 1U) * element_size;
}

void validate_buffer_shapes(const AscendStageArtifact& stage,
                            std::span<const std::uint8_t> x,
                            std::span<const std::uint8_t> mean,
                            std::span<const std::uint8_t> inv_variance,
                            std::span<const std::uint8_t> scale,
                            std::span<const std::uint8_t> bias,
                            std::span<const std::uint8_t> y) {
  const std::size_t xy_size = storage_element_size(stage.output_type());
  if (x.size() != stage.arguments[0].size ||
      y.size() != stage.arguments[5].size || x.size() % xy_size != 0U ||
      y.size() % xy_size != 0U) {
    invalid_oracle("batchnorm X/Y storage size is invalid");
  }
  const std::size_t parameter_bytes =
      static_cast<std::size_t>(stage.batchnorm_channels) * sizeof(float);
  for (const std::span<const std::uint8_t> parameter :
       {mean, inv_variance, scale, bias}) {
    if (parameter.size() != parameter_bytes) {
      invalid_oracle("batchnorm parameter storage size is invalid");
    }
  }
  const std::uint64_t last = static_cast<std::uint64_t>(stage.n_elements - 1);
  (void)load_value(x,
                   stage.input_type(0),
                   tensor_offset(last,
                                 stage.batchnorm_dimensions,
                                 stage.batchnorm_x_strides),
                   "batchnorm X storage is too small");
  const std::uint64_t y_offset = tensor_offset(
      last, stage.batchnorm_dimensions, stage.batchnorm_y_strides);
  if (y_offset >= y.size() / xy_size) {
    invalid_oracle("batchnorm Y storage is too small");
  }
}

}  // namespace

void compute_batchnorm_inference_host_oracle(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> x,
    std::span<const std::uint8_t> mean,
    std::span<const std::uint8_t> inv_variance,
    std::span<const std::uint8_t> scale,
    std::span<const std::uint8_t> bias,
    std::span<std::uint8_t> y) {
  validate_batchnorm_inference_stage_runtime_contract(stage);
  validate_buffer_shapes(stage, x, mean, inv_variance, scale, bias, y);
  validate_output_disjoint(x, mean, inv_variance, scale, bias, y);
  std::vector<bool> written(y.size() / storage_element_size(stage.output_type()),
                            false);
  const auto spatial = static_cast<std::uint64_t>(stage.batchnorm_spatial);
  const auto channels = static_cast<std::uint64_t>(stage.batchnorm_channels);
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const auto logical_index = static_cast<std::uint64_t>(logical);
    const std::uint64_t channel = (logical_index / spatial) % channels;
    const float x_value = load_value(
        x,
        stage.input_type(0),
        tensor_offset(logical_index,
                      stage.batchnorm_dimensions,
                      stage.batchnorm_x_strides),
        "batchnorm X storage is too small");
    const float mean_value = load_value(mean,
                                        StorageDataType::kFloat32,
                                        channel,
                                        "batchnorm mean storage is too small");
    const float inv_value =
        load_value(inv_variance,
                   StorageDataType::kFloat32,
                   channel,
                   "batchnorm inv_variance storage is too small");
    const float scale_value = load_value(
        scale,
        StorageDataType::kFloat32,
        channel,
        "batchnorm scale storage is too small");
    const float bias_value = load_value(bias,
                                        StorageDataType::kFloat32,
                                        channel,
                                        "batchnorm bias storage is too small");
    const float normalized = (x_value - mean_value) * inv_value;
    const float result = normalized * scale_value + bias_value;
    const std::uint64_t output_offset = tensor_offset(
        logical_index, stage.batchnorm_dimensions, stage.batchnorm_y_strides);
    if (written.at(static_cast<std::size_t>(output_offset))) {
      invalid_oracle("batchnorm Y metadata aliases logical elements");
    }
    written[static_cast<std::size_t>(output_offset)] = true;
    store_value(y,
                stage.output_type(),
                output_offset,
                result,
                "batchnorm Y storage is too small");
  }
}

std::size_t batchnorm_inference_kernel_input_count() noexcept { return 5U; }

std::size_t batchnorm_inference_tensor_slot_count() noexcept { return 6U; }

std::size_t batchnorm_inference_output_argument_index() noexcept { return 5U; }

std::size_t batchnorm_inference_runtime_argument_count() noexcept { return 7U; }

const char* batchnorm_inference_output_argument_name() noexcept {
  return "y_ptr";
}

void validate_batchnorm_inference_stage_runtime_contract(
    const AscendStageArtifact& stage) {
  if (stage.kernel_family != KernelFamily::kBatchNormInference ||
      stage.operation != "batchnorm_inference" ||
      stage.input_count != batchnorm_inference_kernel_input_count() ||
      stage.arguments.size() != batchnorm_inference_runtime_argument_count() ||
      stage.tensor_storage_data_types.size() != 6U || stage.n_elements <= 0 ||
      stage.batchnorm_rank < 2 || stage.batchnorm_rank > 8 ||
      stage.batchnorm_channels <= 0 || stage.batchnorm_spatial <= 0 ||
      stage.input_type(0) != stage.output_type() ||
      (stage.output_type() != StorageDataType::kFloat32 &&
       stage.output_type() != StorageDataType::kFloat16 &&
       stage.output_type() != StorageDataType::kBFloat16)) {
    invalid_oracle("batchnorm stage runtime contract is invalid");
  }
  for (std::size_t index = 1; index < 5; ++index) {
    if (stage.input_type(index) != StorageDataType::kFloat32) {
      invalid_oracle("batchnorm runtime parameters must be float32");
    }
  }
  const std::size_t first_axis =
      kMaximumRank - static_cast<std::size_t>(stage.batchnorm_rank);
  std::uint64_t n_elements = 1;
  std::uint64_t spatial = 1;
  for (std::size_t axis = 0; axis < kMaximumRank; ++axis) {
    const std::int64_t dimension = stage.batchnorm_dimensions[axis];
    if (dimension <= 0 || stage.batchnorm_x_strides[axis] < 0 ||
        stage.batchnorm_y_strides[axis] < 0 ||
        (axis < first_axis &&
         (dimension != 1 || stage.batchnorm_x_strides[axis] != 0 ||
          stage.batchnorm_y_strides[axis] != 0))) {
      invalid_oracle("batchnorm runtime tensor metadata is invalid");
    }
    const auto encoded_dimension = static_cast<std::uint64_t>(dimension);
    if (n_elements >
        std::numeric_limits<std::uint64_t>::max() / encoded_dimension) {
      invalid_oracle("batchnorm runtime tensor element count overflows");
    }
    n_elements *= encoded_dimension;
    if (axis > first_axis + 1U) {
      if (spatial >
          std::numeric_limits<std::uint64_t>::max() / encoded_dimension) {
        invalid_oracle("batchnorm runtime spatial size overflows");
      }
      spatial *= encoded_dimension;
    }
  }
  if (n_elements != static_cast<std::uint64_t>(stage.n_elements) ||
      stage.batchnorm_dimensions[first_axis + 1U] !=
          stage.batchnorm_channels ||
      spatial != static_cast<std::uint64_t>(stage.batchnorm_spatial)) {
    invalid_oracle("batchnorm runtime decomposition differs from shape");
  }
  validate_non_overlapping_tensor(stage.batchnorm_dimensions,
                                  stage.batchnorm_x_strides,
                                  first_axis,
                                  "batchnorm X strides alias logical elements");
  validate_non_overlapping_tensor(stage.batchnorm_dimensions,
                                  stage.batchnorm_y_strides,
                                  first_axis,
                                  "batchnorm Y strides alias logical elements");
  validate_pointer_argument(stage.arguments[0], 0U, "x_ptr", false);
  validate_pointer_argument(stage.arguments[1], 1U, "mean_ptr", true);
  validate_pointer_argument(
      stage.arguments[2], 2U, "inv_variance_ptr", true);
  validate_pointer_argument(stage.arguments[3], 3U, "scale_ptr", true);
  validate_pointer_argument(stage.arguments[4], 4U, "bias_ptr", true);
  validate_pointer_argument(stage.arguments[5],
                            batchnorm_inference_output_argument_index(),
                            batchnorm_inference_output_argument_name(),
                            false);
  const std::size_t x_bytes = required_tensor_bytes(stage.batchnorm_dimensions,
                                                    stage.batchnorm_x_strides,
                                                    stage.input_type(0));
  const std::size_t y_bytes = required_tensor_bytes(stage.batchnorm_dimensions,
                                                    stage.batchnorm_y_strides,
                                                    stage.output_type());
  const std::size_t parameter_bytes =
      static_cast<std::size_t>(stage.batchnorm_channels) * sizeof(float);
  const std::array<std::size_t, 6> expected_sizes = {
      x_bytes, parameter_bytes, parameter_bytes,
      parameter_bytes, parameter_bytes, y_bytes};
  for (std::size_t index = 0; index < expected_sizes.size(); ++index) {
    const ArgumentSource& argument = stage.arguments[index];
    if (argument.size != expected_sizes[index] ||
        !valid_alignment(argument.alignment) ||
        (argument.source == ArgumentSourceKind::kBinding &&
         argument.workspace_offset != 0U) ||
        (argument.source == ArgumentSourceKind::kGraphWorkspace &&
         (argument.alignment != kGraphWorkspaceAlignment ||
          argument.workspace_offset % kGraphWorkspaceAlignment != 0U))) {
      invalid_oracle("batchnorm runtime pointer storage metadata is invalid");
    }
  }
  const ArgumentSource& scalar = stage.arguments[6];
  const auto* n_elements_scalar = std::get_if<std::int32_t>(&scalar.scalar);
  if (scalar.index != 6U || scalar.name != "n_elements" ||
      scalar.source != ArgumentSourceKind::kScalar ||
      scalar.type != RawArgumentType::kI32 || n_elements_scalar == nullptr ||
      *n_elements_scalar != stage.n_elements) {
    invalid_oracle("batchnorm runtime scalar ABI is invalid");
  }
  std::set<std::int64_t> pointer_uids;
  for (std::size_t index = 0; index < 6U; ++index) {
    if (stage.arguments[index].uid <= 0 ||
        !pointer_uids.insert(stage.arguments[index].uid).second) {
      invalid_oracle("batchnorm runtime pointer UIDs must be distinct");
    }
  }
}

void seed_batchnorm_inference_host_inputs(
    const AscendStageArtifact& stage,
    std::span<std::uint8_t> x,
    std::span<std::uint8_t> mean,
    std::span<std::uint8_t> inv_variance,
    std::span<std::uint8_t> scale,
    std::span<std::uint8_t> bias) {
  validate_batchnorm_inference_stage_runtime_contract(stage);
  const std::vector<std::uint8_t> output_probe(
      storage_element_size(stage.output_type()) *
          (tensor_offset(static_cast<std::uint64_t>(stage.n_elements - 1),
                         stage.batchnorm_dimensions,
                         stage.batchnorm_y_strides) +
           1U),
      kPaddingSentinel);
  validate_buffer_shapes(
      stage, x, mean, inv_variance, scale, bias, output_probe);
  constexpr std::array<float, 8> kMeans = {
      32.0F, -32.0F, 0.0001F, -0.0001F, 4.0F, -4.0F, 16.0F, -16.0F};
  constexpr std::array<float, 8> kInvVariances = {
      8.0F, 1.0F / 64.0F, 0.125F, 2.0F, 0.5F, 4.0F, 0.25F, 1.0F};
  constexpr std::array<float, 8> kScales = {
      4.0F, -8.0F, 0.5F, -2.0F, 1.25F, -0.75F, 3.0F, -4.0F};
  constexpr std::array<float, 8> kBiases = {
      1.0F, -200.0F, 100.0F, -0.001F, 32.0F, -64.0F, 8.0F, -16.0F};
  for (std::int64_t channel = 0; channel < stage.batchnorm_channels;
       ++channel) {
    const std::size_t code = static_cast<std::size_t>(channel) % kMeans.size();
    store_value(mean,
                StorageDataType::kFloat32,
                static_cast<std::uint64_t>(channel),
                kMeans[code],
                "batchnorm mean storage is too small");
    store_value(inv_variance,
                StorageDataType::kFloat32,
                static_cast<std::uint64_t>(channel),
                kInvVariances[code],
                "batchnorm inv_variance storage is too small");
    store_value(scale,
                StorageDataType::kFloat32,
                static_cast<std::uint64_t>(channel),
                kScales[code],
                "batchnorm scale storage is too small");
    store_value(bias,
                StorageDataType::kFloat32,
                static_cast<std::uint64_t>(channel),
                kBiases[code],
                "batchnorm bias storage is too small");
  }
  const StorageDataType x_type = stage.input_type(0);
  const float cancellation_delta =
      x_type == StorageDataType::kFloat32
          ? 1.0F / 256.0F
          : x_type == StorageDataType::kFloat16 ? 1.0F / 32.0F : 1.0F / 4.0F;
  const auto spatial = static_cast<std::uint64_t>(stage.batchnorm_spatial);
  const auto channels = static_cast<std::uint64_t>(stage.batchnorm_channels);
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const auto logical_index = static_cast<std::uint64_t>(logical);
    const std::uint64_t channel = (logical_index / spatial) % channels;
    const float mean_value = kMeans[static_cast<std::size_t>(channel) % kMeans.size()];
    const float bounded_value =
        static_cast<float>((static_cast<std::uint32_t>(logical) * 13U) % 29U) *
            0.125F -
        1.5F;
    const float signed_delta =
        (logical_index % 2U == 0U ? 1.0F : -1.0F) *
        static_cast<float>(1U + ((logical_index / 2U) % 2U)) *
        cancellation_delta;
    const float value = std::abs(mean_value) >= 16.0F
                            ? mean_value + signed_delta
                            : bounded_value;
    store_value(x,
                x_type,
                tensor_offset(logical_index,
                              stage.batchnorm_dimensions,
                              stage.batchnorm_x_strides),
                value,
                "batchnorm X storage is too small");
  }
}

void validate_batchnorm_inference_host_output(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> x,
    std::span<const std::uint8_t> mean,
    std::span<const std::uint8_t> inv_variance,
    std::span<const std::uint8_t> scale,
    std::span<const std::uint8_t> bias,
    std::span<const std::uint8_t> actual) {
  validate_batchnorm_inference_stage_runtime_contract(stage);
  validate_output_disjoint(
      x, mean, inv_variance, scale, bias, actual);
  const std::size_t size = storage_element_size(stage.output_type());
  if (actual.size() % size != 0U) {
    invalid_oracle("batchnorm output has a partial storage element");
  }
  std::vector<std::uint8_t> expected(actual.size(), kPaddingSentinel);
  compute_batchnorm_inference_host_oracle(
      stage, x, mean, inv_variance, scale, bias, expected);
  std::vector<bool> written(actual.size() / size, false);
  const float absolute_base =
      stage.output_type() == StorageDataType::kFloat32
          ? 1.0e-5F
          : stage.output_type() == StorageDataType::kFloat16 ? 1.0e-3F
                                                             : 1.0e-2F;
  const float relative_tolerance =
      stage.output_type() == StorageDataType::kFloat32
          ? 1.0e-4F
          : stage.output_type() == StorageDataType::kFloat16 ? 5.0e-3F
                                                             : 4.0e-2F;
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const std::uint64_t offset = tensor_offset(
        static_cast<std::uint64_t>(logical),
        stage.batchnorm_dimensions,
        stage.batchnorm_y_strides);
    const float expected_value = load_value(expected,
                                            stage.output_type(),
                                            offset,
                                            "batchnorm expected Y is too small");
    const float actual_value = load_value(actual,
                                          stage.output_type(),
                                          offset,
                                          "batchnorm actual Y is too small");
    const float tolerance =
        absolute_base + relative_tolerance * std::abs(expected_value);
    if (std::abs(actual_value - expected_value) > tolerance) {
      invalid_oracle("batchnorm candidate differs from the host oracle");
    }
    if (written.at(static_cast<std::size_t>(offset))) {
      invalid_oracle("batchnorm Y metadata aliases logical elements");
    }
    written[static_cast<std::size_t>(offset)] = true;
  }
  for (std::size_t element = 0; element < written.size(); ++element) {
    if (written[element]) {
      continue;
    }
    for (std::size_t byte = 0; byte < size; ++byte) {
      if (actual[element * size + byte] != kPaddingSentinel) {
        invalid_oracle("batchnorm candidate modified Y padding");
      }
    }
  }
}

void validate_batchnorm_inference_candidate_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> reference,
    std::span<const std::uint8_t> candidate) {
  validate_batchnorm_inference_stage_runtime_contract(stage);
  const std::size_t element_size = storage_element_size(stage.output_type());
  const std::size_t expected_size = stage.arguments[5].size;
  if (reference.size() != expected_size || candidate.size() != expected_size ||
      expected_size % element_size != 0U) {
    invalid_oracle("batchnorm candidate comparison size is invalid");
  }

  std::vector<bool> written(expected_size / element_size, false);
  const float absolute_tolerance =
      stage.output_type() == StorageDataType::kFloat32
          ? 1.0e-5F
          : stage.output_type() == StorageDataType::kFloat16 ? 1.0e-3F
                                                             : 1.0e-2F;
  const float relative_tolerance =
      stage.output_type() == StorageDataType::kFloat32
          ? 1.0e-4F
          : stage.output_type() == StorageDataType::kFloat16 ? 5.0e-3F
                                                             : 4.0e-2F;
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const std::uint64_t offset = tensor_offset(
        static_cast<std::uint64_t>(logical),
        stage.batchnorm_dimensions,
        stage.batchnorm_y_strides);
    const float reference_value = load_value(
        reference,
        stage.output_type(),
        offset,
        "batchnorm reference candidate Y is too small");
    const float candidate_value = load_value(
        candidate,
        stage.output_type(),
        offset,
        "batchnorm compared candidate Y is too small");
    const float tolerance =
        absolute_tolerance + relative_tolerance * std::abs(reference_value);
    if (std::abs(candidate_value - reference_value) > tolerance) {
      invalid_oracle("batchnorm autotune candidates differ logically");
    }
    if (written.at(static_cast<std::size_t>(offset))) {
      invalid_oracle("batchnorm candidate comparison aliases logical Y");
    }
    written[static_cast<std::size_t>(offset)] = true;
  }
  for (std::size_t element = 0; element < written.size(); ++element) {
    if (written[element]) {
      continue;
    }
    for (std::size_t byte = 0; byte < element_size; ++byte) {
      const std::size_t offset = element * element_size + byte;
      if (reference[offset] != kPaddingSentinel ||
          candidate[offset] != kPaddingSentinel) {
        invalid_oracle("batchnorm autotune candidate padding differs");
      }
    }
  }
}

void commit_batchnorm_inference_host_output(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> actual,
    std::span<std::uint8_t> shadow) {
  validate_batchnorm_inference_stage_runtime_contract(stage);
  const std::size_t expected_size = stage.arguments[5].size;
  if (actual.size() != expected_size || shadow.size() != expected_size) {
    invalid_oracle("batchnorm output commit size is invalid");
  }
  std::copy(actual.begin(), actual.end(), shadow.begin());
}

}  // namespace flagdnn::ascend
