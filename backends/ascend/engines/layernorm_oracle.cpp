/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/layernorm_oracle.hpp"

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
#include <string>
#include <vector>

namespace flagdnn::ascend {
namespace {

constexpr std::size_t kGraphWorkspaceAlignment = 256U;

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
      invalid_oracle("LayerNorm oracle storage data type is invalid");
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
                 std::size_t element,
                 const char* range_error) {
  const std::size_t size = storage_element_size(type);
  if (storage.size() % size != 0U || element >= storage.size() / size) {
    invalid_oracle(range_error);
  }
  float result = 0.0F;
  if (type == StorageDataType::kFloat32) {
    std::memcpy(&result, storage.data() + element * size, sizeof(result));
  } else {
    std::uint16_t encoded = 0U;
    std::memcpy(&encoded, storage.data() + element * size, sizeof(encoded));
    result = type == StorageDataType::kFloat16
                 ? half_to_float(encoded)
                 : std::bit_cast<float>(static_cast<std::uint32_t>(encoded)
                                        << 16U);
  }
  if (!std::isfinite(result)) {
    invalid_oracle("LayerNorm oracle values must be finite");
  }
  return result;
}

void store_value(std::span<std::uint8_t> storage,
                 StorageDataType type,
                 std::size_t element,
                 float value,
                 const char* range_error) {
  if (!std::isfinite(value)) {
    invalid_oracle("LayerNorm oracle result must be finite");
  }
  const std::size_t size = storage_element_size(type);
  if (storage.size() % size != 0U || element >= storage.size() / size) {
    invalid_oracle(range_error);
  }
  if (type == StorageDataType::kFloat32) {
    std::memcpy(storage.data() + element * size, &value, sizeof(value));
    return;
  }
  std::uint16_t encoded = 0U;
  if (type == StorageDataType::kFloat16) {
    encoded = float_to_half(value);
    if ((encoded & 0x7C00U) == 0x7C00U) {
      invalid_oracle("LayerNorm float16 result must be finite");
    }
  } else {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    encoded = static_cast<std::uint16_t>(bits >> 16U);
    if ((encoded & 0x7F80U) == 0x7F80U) {
      invalid_oracle("LayerNorm bfloat16 result must be finite");
    }
  }
  std::memcpy(storage.data() + element * size, &encoded, sizeof(encoded));
}

bool valid_alignment(std::size_t alignment) {
  return alignment != 0U && (alignment & (alignment - 1U)) == 0U;
}

bool spans_overlap(std::span<const std::uint8_t> left,
                   std::span<const std::uint8_t> right) {
  if (left.empty() || right.empty()) {
    return false;
  }
  const auto left_start = reinterpret_cast<std::uintptr_t>(left.data());
  const auto right_start = reinterpret_cast<std::uintptr_t>(right.data());
  if (left_start > std::numeric_limits<std::uintptr_t>::max() - left.size() ||
      right_start >
          std::numeric_limits<std::uintptr_t>::max() - right.size()) {
    invalid_oracle("LayerNorm host buffer address range overflows");
  }
  return left_start < right_start + right.size() &&
         right_start < left_start + left.size();
}

void validate_disjoint_tensors(
    std::span<const std::uint8_t> x,
    std::span<const std::uint8_t> scale,
    std::span<const std::uint8_t> bias,
    std::span<const std::uint8_t> y,
    std::span<const std::uint8_t> mean,
    std::span<const std::uint8_t> inv_variance) {
  const std::array<std::span<const std::uint8_t>, 6> tensors = {
      x, scale, bias, y, mean, inv_variance};
  for (std::size_t left = 0U; left < tensors.size(); ++left) {
    for (std::size_t right = left + 1U; right < tensors.size(); ++right) {
      if (spans_overlap(tensors[left], tensors[right])) {
        invalid_oracle("LayerNorm host tensor storage overlaps");
      }
    }
  }
}

void validate_pointer_argument(const ArgumentSource& argument,
                               std::size_t index,
                               const char* name) {
  if (argument.index != index || argument.name != name ||
      argument.type != RawArgumentType::kPointer ||
      (argument.source != ArgumentSourceKind::kBinding &&
       argument.source != ArgumentSourceKind::kGraphWorkspace)) {
    invalid_oracle("LayerNorm stage runtime pointer ABI is invalid");
  }
}

std::size_t checked_elements(std::int64_t value, const char* message) {
  if (value <= 0 ||
      static_cast<std::uint64_t>(value) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    invalid_oracle(message);
  }
  return static_cast<std::size_t>(value);
}

std::size_t checked_bytes(std::size_t elements,
                          std::size_t element_size,
                          const char* message) {
  if (elements > std::numeric_limits<std::size_t>::max() / element_size) {
    invalid_oracle(message);
  }
  return elements * element_size;
}

void validate_buffer_sizes(const AscendStageArtifact& stage,
                           std::span<const std::uint8_t> x,
                           std::span<const std::uint8_t> scale,
                           std::span<const std::uint8_t> bias,
                           std::span<const std::uint8_t> y,
                           std::span<const std::uint8_t> mean,
                           std::span<const std::uint8_t> inv_variance) {
  if (x.size() != stage.arguments[0U].size ||
      scale.size() != stage.arguments[1U].size ||
      bias.size() != stage.arguments[2U].size ||
      y.size() != stage.arguments[3U].size ||
      mean.size() != stage.arguments[4U].size ||
      inv_variance.size() != stage.arguments[5U].size) {
    invalid_oracle("LayerNorm host storage size is invalid");
  }
}

struct Tolerance {
  float absolute;
  float relative;
};

Tolerance output_tolerance(StorageDataType type) {
  switch (type) {
    case StorageDataType::kFloat32:
      // The NPU kernel accumulates long rows in float32.  Near-zero outputs
      // can therefore differ from the double-accumulation host oracle by a
      // few ulps of the row mean even when the result is numerically valid.
      return {2.0e-4F, 3.0e-4F};
    case StorageDataType::kFloat16:
      return {2.0e-3F, 2.0e-2F};
    case StorageDataType::kBFloat16:
      return {2.0e-2F, 6.0e-2F};
    default:
      invalid_oracle("LayerNorm output tolerance data type is invalid");
  }
}

void require_close(float expected,
                   float actual,
                   Tolerance tolerance,
                   const char* message) {
  if (!std::isfinite(expected) || !std::isfinite(actual) ||
      std::abs(actual - expected) >
          tolerance.absolute + tolerance.relative * std::abs(expected)) {
    throw std::invalid_argument(std::string(message) + " (expected=" +
                                std::to_string(expected) + ", actual=" +
                                std::to_string(actual) + ")");
  }
}

}  // namespace

std::size_t layernorm_kernel_input_count() noexcept { return 3U; }

std::size_t layernorm_tensor_slot_count() noexcept { return 6U; }

std::size_t layernorm_runtime_argument_count() noexcept { return 7U; }

std::size_t layernorm_y_argument_index() noexcept { return 3U; }

std::size_t layernorm_mean_argument_index() noexcept { return 4U; }

std::size_t layernorm_inv_variance_argument_index() noexcept { return 5U; }

void validate_layernorm_stage_runtime_contract(const AscendStageArtifact& stage) {
  if (stage.operation != "layernorm" ||
      stage.kernel_family != KernelFamily::kLayerNorm ||
      stage.input_count != layernorm_kernel_input_count() ||
      stage.tensor_storage_data_types.size() != layernorm_tensor_slot_count() ||
      stage.arguments.size() != layernorm_runtime_argument_count() ||
      stage.layernorm_rows <= 0 ||
      stage.layernorm_normalized_elements <= 0 ||
      !std::isfinite(stage.layernorm_epsilon) ||
      stage.layernorm_epsilon <= 0.0) {
    invalid_oracle("LayerNorm stage runtime contract is invalid");
  }
  const std::int64_t rows = stage.layernorm_rows;
  const std::int64_t normalized = stage.layernorm_normalized_elements;
  if (rows > std::numeric_limits<std::int32_t>::max() / normalized ||
      rows * normalized != stage.n_elements) {
    invalid_oracle("LayerNorm runtime decomposition differs from element count");
  }
  const StorageDataType type = stage.input_type(0U);
  if ((type != StorageDataType::kFloat32 &&
       type != StorageDataType::kFloat16 &&
       type != StorageDataType::kBFloat16) ||
      stage.input_type(1U) != type || stage.input_type(2U) != type ||
      stage.output_type() != type ||
      stage.tensor_storage_data_types[4U] != StorageDataType::kFloat32 ||
      stage.tensor_storage_data_types[5U] != StorageDataType::kFloat32) {
    invalid_oracle("LayerNorm runtime tensor data types are invalid");
  }

  static constexpr std::array<const char*, 6> kNames = {
      "x_ptr", "scale_ptr", "bias_ptr", "y_ptr", "mean_ptr",
      "inv_variance_ptr"};
  for (std::size_t index = 0U; index < kNames.size(); ++index) {
    validate_pointer_argument(stage.arguments[index], index, kNames[index]);
  }

  const std::size_t n_elements =
      checked_elements(stage.n_elements, "LayerNorm element count is invalid");
  const std::size_t normalized_elements = checked_elements(
      normalized, "LayerNorm normalized element count is invalid");
  const std::size_t row_count =
      checked_elements(rows, "LayerNorm row count is invalid");
  const std::size_t element_size = storage_element_size(type);
  const std::vector<std::size_t> expected_sizes = {
      checked_bytes(n_elements, element_size, "LayerNorm X size overflows"),
      checked_bytes(normalized_elements,
                    element_size,
                    "LayerNorm scale size overflows"),
      checked_bytes(normalized_elements,
                    element_size,
                    "LayerNorm bias size overflows"),
      checked_bytes(n_elements, element_size, "LayerNorm Y size overflows"),
      checked_bytes(row_count, sizeof(float), "LayerNorm mean size overflows"),
      checked_bytes(row_count,
                    sizeof(float),
                    "LayerNorm inverse variance size overflows")};
  std::set<std::int64_t> pointer_uids;
  for (std::size_t index = 0U; index < expected_sizes.size(); ++index) {
    const ArgumentSource& argument = stage.arguments[index];
    if (argument.size != expected_sizes[index] ||
        !valid_alignment(argument.alignment) || argument.uid <= 0 ||
        !pointer_uids.insert(argument.uid).second ||
        (argument.source == ArgumentSourceKind::kBinding &&
         argument.workspace_offset != 0U) ||
        (argument.source == ArgumentSourceKind::kGraphWorkspace &&
         (argument.alignment != kGraphWorkspaceAlignment ||
          argument.workspace_offset % kGraphWorkspaceAlignment != 0U))) {
      invalid_oracle("LayerNorm runtime pointer storage metadata is invalid");
    }
  }
  const ArgumentSource& scalar = stage.arguments[6U];
  const auto* n_elements_scalar = std::get_if<std::int32_t>(&scalar.scalar);
  if (scalar.index != 6U || scalar.name != "n_elements" ||
      scalar.source != ArgumentSourceKind::kScalar ||
      scalar.type != RawArgumentType::kI32 || n_elements_scalar == nullptr ||
      *n_elements_scalar != stage.n_elements) {
    invalid_oracle("LayerNorm runtime scalar ABI is invalid");
  }
}

void seed_layernorm_host_inputs(const AscendStageArtifact& stage,
                                std::span<std::uint8_t> x,
                                std::span<std::uint8_t> scale,
                                std::span<std::uint8_t> bias) {
  validate_layernorm_stage_runtime_contract(stage);
  std::vector<std::uint8_t> y(stage.arguments[3U].size, 0U);
  std::vector<std::uint8_t> mean(stage.arguments[4U].size, 0U);
  std::vector<std::uint8_t> inv(stage.arguments[5U].size, 0U);
  validate_buffer_sizes(stage, x, scale, bias, y, mean, inv);
  if (spans_overlap(x, scale) || spans_overlap(x, bias) ||
      spans_overlap(scale, bias)) {
    invalid_oracle("LayerNorm input storage overlaps");
  }
  const StorageDataType type = stage.input_type(0U);
  const float delta = type == StorageDataType::kBFloat16
                          ? 0.25F
                          : type == StorageDataType::kFloat16 ? 0.03125F
                                                              : 0.00390625F;
  const std::size_t normalized =
      static_cast<std::size_t>(stage.layernorm_normalized_elements);
  for (std::int64_t row = 0; row < stage.layernorm_rows; ++row) {
    const float base = row % 2 == 0 ? 32.0F : -32.0F;
    for (std::size_t column = 0U; column < normalized; ++column) {
      const int code = static_cast<int>((column * 7U +
                                         static_cast<std::size_t>(row) * 3U) %
                                        11U) -
                       5;
      store_value(x,
                  type,
                  static_cast<std::size_t>(row) * normalized + column,
                  base + static_cast<float>(code) * delta,
                  "LayerNorm X storage is too small");
    }
  }
  for (std::size_t column = 0U; column < normalized; ++column) {
    const float scale_value =
        (column % 2U == 0U ? 1.0F : -1.0F) *
        (0.5F + static_cast<float>(column % 7U) * 0.125F);
    const float bias_value =
        static_cast<float>((column * 5U) % 17U) * 0.0625F - 0.5F;
    store_value(scale,
                type,
                column,
                scale_value,
                "LayerNorm scale storage is too small");
    store_value(bias,
                type,
                column,
                bias_value,
                "LayerNorm bias storage is too small");
  }
}

void compute_layernorm_host_oracle(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> x,
    std::span<const std::uint8_t> scale,
    std::span<const std::uint8_t> bias,
    std::span<std::uint8_t> y,
    std::span<std::uint8_t> mean,
    std::span<std::uint8_t> inv_variance) {
  validate_layernorm_stage_runtime_contract(stage);
  validate_buffer_sizes(stage, x, scale, bias, y, mean, inv_variance);
  validate_disjoint_tensors(x, scale, bias, y, mean, inv_variance);
  const StorageDataType type = stage.input_type(0U);
  const std::size_t normalized =
      static_cast<std::size_t>(stage.layernorm_normalized_elements);
  const std::size_t rows = static_cast<std::size_t>(stage.layernorm_rows);
  for (std::size_t row = 0U; row < rows; ++row) {
    double sum = 0.0;
    for (std::size_t column = 0U; column < normalized; ++column) {
      sum += load_value(x,
                        type,
                        row * normalized + column,
                        "LayerNorm X storage is too small");
    }
    const float row_mean =
        static_cast<float>(sum / static_cast<double>(normalized));
    double centered_square_sum = 0.0;
    for (std::size_t column = 0U; column < normalized; ++column) {
      const double centered =
          static_cast<double>(load_value(x,
                                         type,
                                         row * normalized + column,
                                         "LayerNorm X storage is too small")) -
          row_mean;
      centered_square_sum += centered * centered;
    }
    const float inverse = static_cast<float>(
        1.0 / std::sqrt(centered_square_sum / static_cast<double>(normalized) +
                        stage.layernorm_epsilon));
    store_value(mean,
                StorageDataType::kFloat32,
                row,
                row_mean,
                "LayerNorm mean storage is too small");
    store_value(inv_variance,
                StorageDataType::kFloat32,
                row,
                inverse,
                "LayerNorm inverse variance storage is too small");
    for (std::size_t column = 0U; column < normalized; ++column) {
      const std::size_t index = row * normalized + column;
      const float result =
          (load_value(x, type, index, "LayerNorm X storage is too small") -
           row_mean) *
              inverse *
              load_value(scale,
                         type,
                         column,
                         "LayerNorm scale storage is too small") +
          load_value(
              bias, type, column, "LayerNorm bias storage is too small");
      store_value(
          y, type, index, result, "LayerNorm Y storage is too small");
    }
  }
}

void validate_layernorm_host_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> x,
    std::span<const std::uint8_t> scale,
    std::span<const std::uint8_t> bias,
    std::span<const std::uint8_t> y,
    std::span<const std::uint8_t> mean,
    std::span<const std::uint8_t> inv_variance) {
  validate_layernorm_stage_runtime_contract(stage);
  validate_buffer_sizes(stage, x, scale, bias, y, mean, inv_variance);
  validate_disjoint_tensors(x, scale, bias, y, mean, inv_variance);
  std::vector<std::uint8_t> expected_y(stage.arguments[3U].size, 0U);
  std::vector<std::uint8_t> expected_mean(stage.arguments[4U].size, 0U);
  std::vector<std::uint8_t> expected_inv(stage.arguments[5U].size, 0U);
  compute_layernorm_host_oracle(
      stage, x, scale, bias, expected_y, expected_mean, expected_inv);
  const StorageDataType type = stage.output_type();
  const Tolerance y_tolerance = output_tolerance(type);
  for (std::int32_t index = 0; index < stage.n_elements; ++index) {
    const std::size_t element = static_cast<std::size_t>(index);
    require_close(load_value(expected_y,
                             type,
                             element,
                             "LayerNorm expected Y is too small"),
                  load_value(y,
                             type,
                             element,
                             "LayerNorm actual Y is too small"),
                  y_tolerance,
                  "LayerNorm candidate Y differs from the host oracle");
  }
  const Tolerance statistic_tolerance = {2.0e-5F, 8.0e-4F};
  for (std::int64_t row = 0; row < stage.layernorm_rows; ++row) {
    const std::size_t element = static_cast<std::size_t>(row);
    require_close(load_value(expected_mean,
                             StorageDataType::kFloat32,
                             element,
                             "LayerNorm expected mean is too small"),
                  load_value(mean,
                             StorageDataType::kFloat32,
                             element,
                             "LayerNorm actual mean is too small"),
                  statistic_tolerance,
                  "LayerNorm mean differs from the host oracle");
    require_close(
        load_value(expected_inv,
                   StorageDataType::kFloat32,
                   element,
                   "LayerNorm expected inverse variance is too small"),
        load_value(inv_variance,
                   StorageDataType::kFloat32,
                   element,
                   "LayerNorm actual inverse variance is too small"),
        statistic_tolerance,
        "LayerNorm inverse variance differs from the host oracle");
  }
}

void validate_layernorm_candidate_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> reference_y,
    std::span<const std::uint8_t> reference_mean,
    std::span<const std::uint8_t> reference_inv_variance,
    std::span<const std::uint8_t> candidate_y,
    std::span<const std::uint8_t> candidate_mean,
    std::span<const std::uint8_t> candidate_inv_variance) {
  validate_layernorm_stage_runtime_contract(stage);
  if (reference_y.size() != stage.arguments[3U].size ||
      candidate_y.size() != stage.arguments[3U].size ||
      reference_mean.size() != stage.arguments[4U].size ||
      candidate_mean.size() != stage.arguments[4U].size ||
      reference_inv_variance.size() != stage.arguments[5U].size ||
      candidate_inv_variance.size() != stage.arguments[5U].size) {
    invalid_oracle("LayerNorm candidate comparison size is invalid");
  }
  const StorageDataType type = stage.output_type();
  const Tolerance y_tolerance = output_tolerance(type);
  for (std::int32_t index = 0; index < stage.n_elements; ++index) {
    const std::size_t element = static_cast<std::size_t>(index);
    require_close(load_value(reference_y,
                             type,
                             element,
                             "LayerNorm reference Y is too small"),
                  load_value(candidate_y,
                             type,
                             element,
                             "LayerNorm candidate Y is too small"),
                  y_tolerance,
                  "LayerNorm autotune candidate Y differs logically");
  }
  const Tolerance statistic_tolerance = {2.0e-5F, 8.0e-4F};
  for (std::int64_t row = 0; row < stage.layernorm_rows; ++row) {
    const std::size_t element = static_cast<std::size_t>(row);
    require_close(load_value(reference_mean,
                             StorageDataType::kFloat32,
                             element,
                             "LayerNorm reference mean is too small"),
                  load_value(candidate_mean,
                             StorageDataType::kFloat32,
                             element,
                             "LayerNorm candidate mean is too small"),
                  statistic_tolerance,
                  "LayerNorm autotune mean differs logically");
    require_close(
        load_value(reference_inv_variance,
                   StorageDataType::kFloat32,
                   element,
                   "LayerNorm reference inverse variance is too small"),
        load_value(candidate_inv_variance,
                   StorageDataType::kFloat32,
                   element,
                   "LayerNorm candidate inverse variance is too small"),
        statistic_tolerance,
        "LayerNorm autotune inverse variance differs logically");
  }
}

void commit_layernorm_host_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> actual_y,
    std::span<const std::uint8_t> actual_mean,
    std::span<const std::uint8_t> actual_inv_variance,
    std::span<std::uint8_t> shadow_y,
    std::span<std::uint8_t> shadow_mean,
    std::span<std::uint8_t> shadow_inv_variance) {
  validate_layernorm_stage_runtime_contract(stage);
  if (actual_y.size() != stage.arguments[3U].size ||
      shadow_y.size() != stage.arguments[3U].size ||
      actual_mean.size() != stage.arguments[4U].size ||
      shadow_mean.size() != stage.arguments[4U].size ||
      actual_inv_variance.size() != stage.arguments[5U].size ||
      shadow_inv_variance.size() != stage.arguments[5U].size) {
    invalid_oracle("LayerNorm output commit size is invalid");
  }
  std::copy(actual_y.begin(), actual_y.end(), shadow_y.begin());
  std::copy(actual_mean.begin(), actual_mean.end(), shadow_mean.begin());
  std::copy(actual_inv_variance.begin(),
            actual_inv_variance.end(),
            shadow_inv_variance.begin());
}

}  // namespace flagdnn::ascend
