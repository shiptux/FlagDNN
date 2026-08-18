/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/rmsnorm_oracle.hpp"

#include <algorithm>
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
      invalid_oracle("RMSNorm oracle storage data type is invalid");
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
    invalid_oracle("RMSNorm oracle values must be finite");
  }
  return result;
}

void store_value(std::span<std::uint8_t> storage,
                 StorageDataType type,
                 std::size_t element,
                 float value,
                 const char* range_error) {
  if (!std::isfinite(value)) {
    invalid_oracle("RMSNorm oracle result must be finite");
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
      invalid_oracle("RMSNorm float16 result must be finite");
    }
  } else {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    encoded = static_cast<std::uint16_t>(bits >> 16U);
    if ((encoded & 0x7F80U) == 0x7F80U) {
      invalid_oracle("RMSNorm bfloat16 result must be finite");
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
    invalid_oracle("RMSNorm host buffer address range overflows");
  }
  const std::uintptr_t left_end = left_start + left.size();
  const std::uintptr_t right_end = right_start + right.size();
  return left_start < right_end && right_start < left_end;
}

void validate_disjoint_outputs(std::span<const std::uint8_t> x,
                               std::span<const std::uint8_t> scale,
                               std::span<const std::uint8_t> bias,
                               std::span<const std::uint8_t> y,
                               std::span<const std::uint8_t> inv_variance) {
  for (const std::span<const std::uint8_t> input : {x, scale, bias}) {
    if (spans_overlap(input, y) || spans_overlap(input, inv_variance)) {
      invalid_oracle("RMSNorm output storage overlaps an input");
    }
  }
  if (spans_overlap(y, inv_variance)) {
    invalid_oracle("RMSNorm output storage overlaps another output");
  }
}

void validate_pointer_argument(const ArgumentSource& argument,
                               std::size_t index,
                               const char* name) {
  if (argument.index != index || argument.name != name ||
      argument.type != RawArgumentType::kPointer ||
      (argument.source != ArgumentSourceKind::kBinding &&
       argument.source != ArgumentSourceKind::kGraphWorkspace)) {
    invalid_oracle("RMSNorm stage runtime pointer ABI is invalid");
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
                           std::span<const std::uint8_t> inv_variance) {
  if (x.size() != stage.arguments[0].size ||
      scale.size() != stage.arguments[1].size ||
      bias.size() != stage.arguments[2].size ||
      y.size() != stage.arguments[3].size ||
      inv_variance.size() != stage.arguments[4].size) {
    invalid_oracle("RMSNorm host storage size is invalid");
  }
}

struct Tolerance {
  float absolute;
  float relative;
};

Tolerance output_tolerance(StorageDataType type) {
  switch (type) {
    case StorageDataType::kFloat32:
      return {1.0e-5F, 2.0e-4F};
    case StorageDataType::kFloat16:
      return {1.0e-3F, 1.0e-2F};
    case StorageDataType::kBFloat16:
      return {1.0e-2F, 5.0e-2F};
    default:
      invalid_oracle("RMSNorm output tolerance data type is invalid");
  }
}

void require_close(float expected,
                   float actual,
                   Tolerance tolerance,
                   const char* message) {
  if (!std::isfinite(expected) || !std::isfinite(actual) ||
      std::abs(actual - expected) >
          tolerance.absolute + tolerance.relative * std::abs(expected)) {
    invalid_oracle(message);
  }
}

}  // namespace

std::size_t rmsnorm_kernel_input_count() noexcept { return 3U; }

std::size_t rmsnorm_tensor_slot_count() noexcept { return 5U; }

std::size_t rmsnorm_runtime_argument_count() noexcept { return 6U; }

std::size_t rmsnorm_y_argument_index() noexcept { return 3U; }

std::size_t rmsnorm_inv_variance_argument_index() noexcept { return 4U; }

void validate_rmsnorm_stage_runtime_contract(const AscendStageArtifact& stage) {
  if (stage.operation != "rmsnorm" ||
      stage.kernel_family != KernelFamily::kRmsNorm ||
      stage.input_count != rmsnorm_kernel_input_count() ||
      stage.tensor_storage_data_types.size() != rmsnorm_tensor_slot_count() ||
      stage.arguments.size() != rmsnorm_runtime_argument_count() ||
      stage.rmsnorm_rows <= 0 || stage.rmsnorm_normalized_elements <= 0 ||
      !std::isfinite(stage.rmsnorm_epsilon) || stage.rmsnorm_epsilon <= 0.0) {
    invalid_oracle("RMSNorm stage runtime contract is invalid");
  }
  const std::int64_t rows = stage.rmsnorm_rows;
  const std::int64_t normalized = stage.rmsnorm_normalized_elements;
  if (rows > std::numeric_limits<std::int32_t>::max() / normalized ||
      rows * normalized != stage.n_elements) {
    invalid_oracle("RMSNorm runtime decomposition differs from element count");
  }
  const StorageDataType type = stage.input_type(0U);
  if ((type != StorageDataType::kFloat32 &&
       type != StorageDataType::kFloat16 &&
       type != StorageDataType::kBFloat16) ||
      stage.input_type(1U) != type || stage.input_type(2U) != type ||
      stage.output_type() != type ||
      stage.tensor_storage_data_types[4U] != StorageDataType::kFloat32) {
    invalid_oracle("RMSNorm runtime tensor data types are invalid");
  }

  validate_pointer_argument(stage.arguments[0U], 0U, "x_ptr");
  validate_pointer_argument(stage.arguments[1U], 1U, "scale_ptr");
  validate_pointer_argument(stage.arguments[2U], 2U, "bias_ptr");
  validate_pointer_argument(stage.arguments[3U], 3U, "y_ptr");
  validate_pointer_argument(
      stage.arguments[4U], 4U, "inv_variance_ptr");

  const std::size_t n_elements = checked_elements(
      stage.n_elements, "RMSNorm element count is invalid");
  const std::size_t normalized_elements = checked_elements(
      normalized, "RMSNorm normalized element count is invalid");
  const std::size_t row_count =
      checked_elements(rows, "RMSNorm row count is invalid");
  const std::size_t element_size = storage_element_size(type);
  const std::vector<std::size_t> expected_sizes = {
      checked_bytes(n_elements, element_size, "RMSNorm X size overflows"),
      checked_bytes(normalized_elements,
                    element_size,
                    "RMSNorm scale size overflows"),
      checked_bytes(normalized_elements,
                    element_size,
                    "RMSNorm bias size overflows"),
      checked_bytes(n_elements, element_size, "RMSNorm Y size overflows"),
      checked_bytes(row_count,
                    sizeof(float),
                    "RMSNorm inverse variance size overflows")};
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
      invalid_oracle("RMSNorm runtime pointer storage metadata is invalid");
    }
  }
  const ArgumentSource& scalar = stage.arguments[5U];
  const auto* n_elements_scalar = std::get_if<std::int32_t>(&scalar.scalar);
  if (scalar.index != 5U || scalar.name != "n_elements" ||
      scalar.source != ArgumentSourceKind::kScalar ||
      scalar.type != RawArgumentType::kI32 || n_elements_scalar == nullptr ||
      *n_elements_scalar != stage.n_elements) {
    invalid_oracle("RMSNorm runtime scalar ABI is invalid");
  }
}

void seed_rmsnorm_host_inputs(const AscendStageArtifact& stage,
                              std::span<std::uint8_t> x,
                              std::span<std::uint8_t> scale,
                              std::span<std::uint8_t> bias) {
  validate_rmsnorm_stage_runtime_contract(stage);
  std::vector<std::uint8_t> y(stage.arguments[3U].size, 0U);
  std::vector<std::uint8_t> inv_variance(stage.arguments[4U].size, 0U);
  validate_buffer_sizes(stage, x, scale, bias, y, inv_variance);
  const StorageDataType type = stage.input_type(0U);
  for (std::int32_t index = 0; index < stage.n_elements; ++index) {
    const float value =
        static_cast<float>((static_cast<std::uint32_t>(index) * 17U) % 37U) *
            0.125F -
        2.25F;
    store_value(x,
                type,
                static_cast<std::size_t>(index),
                value,
                "RMSNorm X storage is too small");
  }
  for (std::int64_t index = 0; index < stage.rmsnorm_normalized_elements;
       ++index) {
    const float scale_value =
        (index % 2 == 0 ? 1.0F : -1.0F) *
        (0.5F + static_cast<float>(index % 7) * 0.125F);
    const float bias_value =
        static_cast<float>((static_cast<std::uint64_t>(index) * 5U) % 17U) *
            0.0625F -
        0.5F;
    store_value(scale,
                type,
                static_cast<std::size_t>(index),
                scale_value,
                "RMSNorm scale storage is too small");
    store_value(bias,
                type,
                static_cast<std::size_t>(index),
                bias_value,
                "RMSNorm bias storage is too small");
  }
}

void compute_rmsnorm_host_oracle(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> x,
    std::span<const std::uint8_t> scale,
    std::span<const std::uint8_t> bias,
    std::span<std::uint8_t> y,
    std::span<std::uint8_t> inv_variance) {
  validate_rmsnorm_stage_runtime_contract(stage);
  validate_buffer_sizes(stage, x, scale, bias, y, inv_variance);
  validate_disjoint_outputs(x, scale, bias, y, inv_variance);
  const StorageDataType type = stage.input_type(0U);
  const std::size_t normalized =
      static_cast<std::size_t>(stage.rmsnorm_normalized_elements);
  const std::size_t rows = static_cast<std::size_t>(stage.rmsnorm_rows);
  for (std::size_t row = 0U; row < rows; ++row) {
    double square_sum = 0.0;
    for (std::size_t column = 0U; column < normalized; ++column) {
      const float value = load_value(x,
                                     type,
                                     row * normalized + column,
                                     "RMSNorm X storage is too small");
      square_sum += static_cast<double>(value) * value;
    }
    const double mean_square = square_sum / static_cast<double>(normalized);
    const float inverse = static_cast<float>(
        1.0 / std::sqrt(mean_square + stage.rmsnorm_epsilon));
    store_value(inv_variance,
                StorageDataType::kFloat32,
                row,
                inverse,
                "RMSNorm inverse variance storage is too small");
    for (std::size_t column = 0U; column < normalized; ++column) {
      const std::size_t index = row * normalized + column;
      const float result =
          load_value(x, type, index, "RMSNorm X storage is too small") *
              inverse *
              load_value(scale,
                         type,
                         column,
                         "RMSNorm scale storage is too small") +
          load_value(
              bias, type, column, "RMSNorm bias storage is too small");
      store_value(
          y, type, index, result, "RMSNorm Y storage is too small");
    }
  }
}

void validate_rmsnorm_host_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> x,
    std::span<const std::uint8_t> scale,
    std::span<const std::uint8_t> bias,
    std::span<const std::uint8_t> y,
    std::span<const std::uint8_t> inv_variance) {
  validate_rmsnorm_stage_runtime_contract(stage);
  validate_buffer_sizes(stage, x, scale, bias, y, inv_variance);
  validate_disjoint_outputs(x, scale, bias, y, inv_variance);
  std::vector<std::uint8_t> expected_y(stage.arguments[3U].size, 0U);
  std::vector<std::uint8_t> expected_inv(stage.arguments[4U].size, 0U);
  compute_rmsnorm_host_oracle(
      stage, x, scale, bias, expected_y, expected_inv);
  const StorageDataType type = stage.output_type();
  const Tolerance y_tolerance = output_tolerance(type);
  for (std::int32_t index = 0; index < stage.n_elements; ++index) {
    const std::size_t element = static_cast<std::size_t>(index);
    require_close(load_value(expected_y,
                             type,
                             element,
                             "RMSNorm expected Y is too small"),
                  load_value(y,
                             type,
                             element,
                             "RMSNorm actual Y is too small"),
                  y_tolerance,
                  "RMSNorm candidate Y differs from the host oracle");
  }
  const Tolerance inv_tolerance = {1.0e-5F, 5.0e-4F};
  for (std::int64_t row = 0; row < stage.rmsnorm_rows; ++row) {
    const std::size_t element = static_cast<std::size_t>(row);
    require_close(load_value(expected_inv,
                             StorageDataType::kFloat32,
                             element,
                             "RMSNorm expected inverse variance is too small"),
                  load_value(inv_variance,
                             StorageDataType::kFloat32,
                             element,
                             "RMSNorm actual inverse variance is too small"),
                  inv_tolerance,
                  "RMSNorm inverse variance differs from the host oracle");
  }
}

void validate_rmsnorm_candidate_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> reference_y,
    std::span<const std::uint8_t> reference_inv_variance,
    std::span<const std::uint8_t> candidate_y,
    std::span<const std::uint8_t> candidate_inv_variance) {
  validate_rmsnorm_stage_runtime_contract(stage);
  if (reference_y.size() != stage.arguments[3U].size ||
      candidate_y.size() != stage.arguments[3U].size ||
      reference_inv_variance.size() != stage.arguments[4U].size ||
      candidate_inv_variance.size() != stage.arguments[4U].size) {
    invalid_oracle("RMSNorm candidate comparison size is invalid");
  }
  const StorageDataType type = stage.output_type();
  const Tolerance y_tolerance = output_tolerance(type);
  for (std::int32_t index = 0; index < stage.n_elements; ++index) {
    const std::size_t element = static_cast<std::size_t>(index);
    require_close(load_value(reference_y,
                             type,
                             element,
                             "RMSNorm reference Y is too small"),
                  load_value(candidate_y,
                             type,
                             element,
                             "RMSNorm candidate Y is too small"),
                  y_tolerance,
                  "RMSNorm autotune candidate Y differs logically");
  }
  const Tolerance inv_tolerance = {1.0e-5F, 5.0e-4F};
  for (std::int64_t row = 0; row < stage.rmsnorm_rows; ++row) {
    const std::size_t element = static_cast<std::size_t>(row);
    require_close(
        load_value(reference_inv_variance,
                   StorageDataType::kFloat32,
                   element,
                   "RMSNorm reference inverse variance is too small"),
        load_value(candidate_inv_variance,
                   StorageDataType::kFloat32,
                   element,
                   "RMSNorm candidate inverse variance is too small"),
        inv_tolerance,
        "RMSNorm autotune inverse variance differs logically");
  }
}

void commit_rmsnorm_host_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> actual_y,
    std::span<const std::uint8_t> actual_inv_variance,
    std::span<std::uint8_t> shadow_y,
    std::span<std::uint8_t> shadow_inv_variance) {
  validate_rmsnorm_stage_runtime_contract(stage);
  if (actual_y.size() != stage.arguments[3U].size ||
      shadow_y.size() != stage.arguments[3U].size ||
      actual_inv_variance.size() != stage.arguments[4U].size ||
      shadow_inv_variance.size() != stage.arguments[4U].size) {
    invalid_oracle("RMSNorm output commit size is invalid");
  }
  std::copy(actual_y.begin(), actual_y.end(), shadow_y.begin());
  std::copy(actual_inv_variance.begin(),
            actual_inv_variance.end(),
            shadow_inv_variance.begin());
}

}  // namespace flagdnn::ascend
