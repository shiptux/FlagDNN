/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/matmul_oracle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::ascend {
namespace {

constexpr std::uint64_t kFullOracleOutputLimit = 65536U;
constexpr std::uint64_t kLargeOracleSampleCount = 257U;

[[noreturn]] void invalid_oracle(std::string_view message) {
  throw std::invalid_argument(std::string(message));
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
  invalid_oracle("MatMul storage data type is invalid");
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

std::uint64_t batch_offset(
    const AscendStageArtifact& stage,
    std::uint64_t batch,
    const std::array<std::int64_t, 6>& strides) {
  std::uint64_t result = 0;
  for (std::size_t reversed = stage.matmul_batch_dimensions.size();
       reversed != 0U;
       --reversed) {
    const std::size_t axis = reversed - 1U;
    const std::int64_t dimension = stage.matmul_batch_dimensions[axis];
    const std::int64_t stride = strides[axis];
    if (dimension <= 0 || stride < 0) {
      invalid_oracle("MatMul batch metadata is invalid");
    }
    const auto unsigned_dimension = static_cast<std::uint64_t>(dimension);
    const std::uint64_t coordinate = batch % unsigned_dimension;
    batch /= unsigned_dimension;
    result = checked_add(
        result,
        checked_multiply(coordinate,
                         static_cast<std::uint64_t>(stride),
                         "MatMul batch offset overflows"),
        "MatMul batch offset overflows");
  }
  if (batch != 0U) {
    invalid_oracle("MatMul batch index exceeds its shape");
  }
  return result;
}

std::uint64_t tensor_offset(
    const AscendStageArtifact& stage,
    std::uint64_t batch,
    std::uint64_t row,
    std::uint64_t column,
    const std::array<std::int64_t, 6>& batch_strides,
    std::int64_t row_stride,
    std::int64_t column_stride) {
  if (row_stride <= 0 || column_stride <= 0) {
    invalid_oracle("MatMul matrix strides are invalid");
  }
  std::uint64_t result = batch_offset(stage, batch, batch_strides);
  result = checked_add(
      result,
      checked_multiply(row,
                       static_cast<std::uint64_t>(row_stride),
                       "MatMul tensor offset overflows"),
      "MatMul tensor offset overflows");
  return checked_add(
      result,
      checked_multiply(column,
                       static_cast<std::uint64_t>(column_stride),
                       "MatMul tensor offset overflows"),
      "MatMul tensor offset overflows");
}

bool matrix_offsets_are_unique(std::int64_t rows,
                               std::int64_t columns,
                               std::int64_t row_stride,
                               std::int64_t column_stride) {
  if (rows <= 0 || columns <= 0 || row_stride <= 0 || column_stride <= 0) {
    return false;
  }
  const std::int64_t divisor = std::gcd(row_stride, column_stride);
  return column_stride / divisor >= rows || row_stride / divisor >= columns;
}

std::uint64_t storage_elements(
    const AscendStageArtifact& stage,
    const std::array<std::int64_t, 6>& batch_strides,
    std::int64_t rows,
    std::int64_t columns,
    std::int64_t row_stride,
    std::int64_t column_stride) {
  if (rows <= 0 || columns <= 0 || row_stride <= 0 || column_stride <= 0) {
    invalid_oracle("MatMul tensor metadata is invalid");
  }
  std::uint64_t maximum = 0;
  for (std::size_t axis = 0; axis < stage.matmul_batch_dimensions.size();
       ++axis) {
    const std::int64_t dimension = stage.matmul_batch_dimensions[axis];
    const std::int64_t stride = batch_strides[axis];
    if (dimension <= 0 || stride < 0) {
      invalid_oracle("MatMul batch metadata is invalid");
    }
    maximum = checked_add(
        maximum,
        checked_multiply(
            static_cast<std::uint64_t>(dimension - 1),
            static_cast<std::uint64_t>(stride),
            "MatMul storage span overflows"),
        "MatMul storage span overflows");
  }
  maximum = checked_add(
      maximum,
      checked_multiply(static_cast<std::uint64_t>(rows - 1),
                       static_cast<std::uint64_t>(row_stride),
                       "MatMul storage span overflows"),
      "MatMul storage span overflows");
  maximum = checked_add(
      maximum,
      checked_multiply(static_cast<std::uint64_t>(columns - 1),
                       static_cast<std::uint64_t>(column_stride),
                       "MatMul storage span overflows"),
      "MatMul storage span overflows");
  return checked_add(maximum, 1U, "MatMul storage span overflows");
}

std::size_t storage_bytes(std::uint64_t elements, StorageDataType type) {
  const std::uint64_t result = checked_multiply(
      elements,
      static_cast<std::uint64_t>(storage_element_size(type)),
      "MatMul storage byte count overflows");
  if (result > std::numeric_limits<std::size_t>::max()) {
    invalid_oracle("MatMul storage byte count exceeds size_t");
  }
  return static_cast<std::size_t>(result);
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
    invalid_oracle("MatMul input offset is out of range");
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
    invalid_oracle("MatMul values must be finite");
  }
  return result;
}

void store_value(std::span<std::uint8_t> bytes,
                 StorageDataType type,
                 std::uint64_t element,
                 float value) {
  if (!std::isfinite(value)) {
    invalid_oracle("MatMul result must be finite");
  }
  const std::size_t size = storage_element_size(type);
  if (element >= bytes.size() / size) {
    invalid_oracle("MatMul output offset is out of range");
  }
  if (type == StorageDataType::kFloat32) {
    std::memcpy(bytes.data() + element * size, &value, sizeof(value));
    return;
  }
  std::uint16_t encoded = 0;
  if (type == StorageDataType::kFloat16) {
    encoded = float_to_half(value);
    if ((encoded & 0x7c00U) == 0x7c00U) {
      invalid_oracle("MatMul result must be finite");
    }
  } else {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x00007fffU + ((bits >> 16U) & 1U);
    encoded = static_cast<std::uint16_t>(bits >> 16U);
    if ((encoded & 0x7f80U) == 0x7f80U) {
      invalid_oracle("MatMul result must be finite");
    }
  }
  std::memcpy(bytes.data() + element * size, &encoded, sizeof(encoded));
}

bool spans_overlap(std::span<const std::uint8_t> left,
                   std::span<const std::uint8_t> right) {
  if (left.empty() || right.empty()) {
    return false;
  }
  const std::uintptr_t left_begin =
      reinterpret_cast<std::uintptr_t>(left.data());
  const std::uintptr_t right_begin =
      reinterpret_cast<std::uintptr_t>(right.data());
  if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left.size() ||
      right_begin > std::numeric_limits<std::uintptr_t>::max() - right.size()) {
    invalid_oracle("MatMul host span address overflows");
  }
  const std::uintptr_t left_end = left_begin + left.size();
  const std::uintptr_t right_end = right_begin + right.size();
  return left_begin < right_end && right_begin < left_end;
}

void validate_host_spans(const AscendStageArtifact& stage,
                         std::span<const std::uint8_t> a,
                         std::span<const std::uint8_t> b,
                         std::span<const std::uint8_t> output) {
  if (a.size() != stage.arguments[0].size ||
      b.size() != stage.arguments[1].size ||
      output.size() != stage.arguments[2].size) {
    invalid_oracle("MatMul host storage size is invalid");
  }
  if (spans_overlap(a, output) || spans_overlap(b, output)) {
    invalid_oracle("MatMul output aliases an input");
  }
}

float expected_value(const AscendStageArtifact& stage,
                     std::span<const std::uint8_t> a,
                     std::span<const std::uint8_t> b,
                     std::uint64_t batch,
                     std::uint64_t row,
                     std::uint64_t column) {
  double accumulator = 0.0;
  for (std::uint64_t reduction = 0;
       reduction < static_cast<std::uint64_t>(stage.matmul_k);
       ++reduction) {
    const std::uint64_t a_offset = tensor_offset(
        stage,
        batch,
        row,
        reduction,
        stage.matmul_a_batch_strides,
        stage.matmul_a_stride_m,
        stage.matmul_a_stride_k);
    const std::uint64_t b_offset = tensor_offset(
        stage,
        batch,
        reduction,
        column,
        stage.matmul_b_batch_strides,
        stage.matmul_b_stride_k,
        stage.matmul_b_stride_n);
    accumulator += static_cast<double>(load_value(a, stage.input_type(0), a_offset)) *
                   static_cast<double>(load_value(b, stage.input_type(1), b_offset));
  }
  if (!std::isfinite(accumulator) ||
      accumulator > static_cast<double>(std::numeric_limits<float>::max()) ||
      accumulator < -static_cast<double>(std::numeric_limits<float>::max())) {
    invalid_oracle("MatMul accumulation must be finite");
  }
  return static_cast<float>(accumulator);
}

std::array<std::uint64_t, 3> output_coordinates(
    const AscendStageArtifact& stage,
    std::uint64_t logical) {
  const auto n = static_cast<std::uint64_t>(stage.matmul_n);
  const auto m = static_cast<std::uint64_t>(stage.matmul_m);
  const std::uint64_t column = logical % n;
  logical /= n;
  const std::uint64_t row = logical % m;
  return {logical / m, row, column};
}

float absolute_tolerance(const AscendStageArtifact& stage) {
  if (stage.output_type() == StorageDataType::kFloat16) {
    return 5.0e-2F;
  }
  if (stage.output_type() == StorageDataType::kBFloat16) {
    return 1.0e-1F;
  }
  return 5.0e-3F *
         std::sqrt(std::max(
             1.0F, static_cast<float>(stage.matmul_k) / 512.0F));
}

float relative_tolerance(const AscendStageArtifact& stage) {
  return stage.output_type() == StorageDataType::kFloat32 ? 5.0e-3F : 5.0e-2F;
}

void validate_one_output(const AscendStageArtifact& stage,
                         std::span<const std::uint8_t> a,
                         std::span<const std::uint8_t> b,
                         std::span<const std::uint8_t> actual,
                         std::uint64_t logical) {
  const auto coordinates = output_coordinates(stage, logical);
  const std::uint64_t offset = tensor_offset(
      stage,
      coordinates[0],
      coordinates[1],
      coordinates[2],
      stage.matmul_output_batch_strides,
      stage.matmul_output_stride_m,
      stage.matmul_output_stride_n);
  const float expected = expected_value(
      stage, a, b, coordinates[0], coordinates[1], coordinates[2]);
  const float observed = load_value(actual, stage.output_type(), offset);
  const float tolerance = absolute_tolerance(stage) +
                          relative_tolerance(stage) * std::abs(expected);
  if (std::abs(observed - expected) > tolerance) {
    invalid_oracle(
        "MatMul output differs from the host oracle at batch=" +
        std::to_string(coordinates[0]) + " row=" +
        std::to_string(coordinates[1]) + " column=" +
        std::to_string(coordinates[2]) + " offset=" +
        std::to_string(offset) + " expected=" + std::to_string(expected) +
        " observed=" + std::to_string(observed) +
        " tolerance=" + std::to_string(tolerance));
  }
}

}  // namespace

std::size_t matmul_kernel_input_count() noexcept { return 2U; }

std::size_t matmul_output_argument_index() noexcept { return 2U; }

std::size_t matmul_tensor_slot_count() noexcept { return 3U; }

std::size_t matmul_runtime_argument_count() noexcept { return 4U; }

const char* matmul_output_argument_name() noexcept { return "output_ptr"; }

void validate_matmul_stage_runtime_contract(const AscendStageArtifact& stage) {
  if (stage.kernel_family != KernelFamily::kMatMul ||
      stage.operation != "matmul" ||
      stage.input_count != matmul_kernel_input_count() ||
      stage.tensor_storage_data_types.size() != matmul_tensor_slot_count() ||
      stage.arguments.size() != matmul_runtime_argument_count() ||
      stage.input_type(0) != stage.input_type(1) ||
      stage.input_type(0) != stage.output_type() ||
      (stage.output_type() != StorageDataType::kFloat32 &&
       stage.output_type() != StorageDataType::kFloat16 &&
       stage.output_type() != StorageDataType::kBFloat16) ||
      stage.matmul_batch <= 0 || stage.matmul_m <= 0 ||
      stage.matmul_n <= 0 || stage.matmul_k <= 0 ||
      !matrix_offsets_are_unique(stage.matmul_m,
                                 stage.matmul_k,
                                 stage.matmul_a_stride_m,
                                 stage.matmul_a_stride_k) ||
      !matrix_offsets_are_unique(stage.matmul_k,
                                 stage.matmul_n,
                                 stage.matmul_b_stride_k,
                                 stage.matmul_b_stride_n) ||
      !matrix_offsets_are_unique(stage.matmul_m,
                                 stage.matmul_n,
                                 stage.matmul_output_stride_m,
                                 stage.matmul_output_stride_n)) {
    invalid_oracle("MatMul stage runtime contract is invalid");
  }
  std::uint64_t batch = 1U;
  for (const std::int64_t dimension : stage.matmul_batch_dimensions) {
    if (dimension <= 0) {
      invalid_oracle("MatMul batch dimensions are invalid");
    }
    batch = checked_multiply(batch,
                             static_cast<std::uint64_t>(dimension),
                             "MatMul batch size overflows");
  }
  const std::uint64_t output_elements = checked_multiply(
      checked_multiply(batch,
                       static_cast<std::uint64_t>(stage.matmul_m),
                       "MatMul output size overflows"),
      static_cast<std::uint64_t>(stage.matmul_n),
      "MatMul output size overflows");
  if (batch != static_cast<std::uint64_t>(stage.matmul_batch) ||
      output_elements != static_cast<std::uint64_t>(stage.n_elements)) {
    invalid_oracle("MatMul decomposition is inconsistent");
  }
  constexpr std::array<const char*, 3> kPointerNames = {
      "a_ptr", "b_ptr", "output_ptr"};
  for (std::size_t index = 0; index < kPointerNames.size(); ++index) {
    const ArgumentSource& argument = stage.arguments[index];
    if (argument.index != index || argument.name != kPointerNames[index] ||
        argument.type != RawArgumentType::kPointer ||
        (argument.source != ArgumentSourceKind::kBinding &&
         argument.source != ArgumentSourceKind::kGraphWorkspace) ||
        argument.uid <= 0 || argument.alignment == 0U ||
        (argument.source == ArgumentSourceKind::kBinding &&
         argument.workspace_offset != 0U) ||
        (argument.source == ArgumentSourceKind::kGraphWorkspace &&
         argument.workspace_offset % argument.alignment != 0U)) {
      invalid_oracle("MatMul pointer ABI is invalid");
    }
  }
  if (stage.arguments[0].uid == stage.arguments[1].uid ||
      stage.arguments[0].uid == stage.arguments[2].uid ||
      stage.arguments[1].uid == stage.arguments[2].uid) {
    invalid_oracle("MatMul tensor UIDs must be distinct");
  }
  const ArgumentSource& scalar = stage.arguments[3];
  const auto* scalar_value = std::get_if<std::int32_t>(&scalar.scalar);
  if (scalar.index != 3U || scalar.name != "n_elements" ||
      scalar.source != ArgumentSourceKind::kScalar ||
      scalar.type != RawArgumentType::kI32 || scalar_value == nullptr ||
      *scalar_value != stage.n_elements) {
    invalid_oracle("MatMul scalar ABI is invalid");
  }
  const std::array<std::uint64_t, 3> expected_elements = {
      storage_elements(stage,
                       stage.matmul_a_batch_strides,
                       stage.matmul_m,
                       stage.matmul_k,
                       stage.matmul_a_stride_m,
                       stage.matmul_a_stride_k),
      storage_elements(stage,
                       stage.matmul_b_batch_strides,
                       stage.matmul_k,
                       stage.matmul_n,
                       stage.matmul_b_stride_k,
                       stage.matmul_b_stride_n),
      storage_elements(stage,
                       stage.matmul_output_batch_strides,
                       stage.matmul_m,
                       stage.matmul_n,
                       stage.matmul_output_stride_m,
                       stage.matmul_output_stride_n),
  };
  for (std::size_t index = 0; index < expected_elements.size(); ++index) {
    if (stage.arguments[index].size !=
        storage_bytes(expected_elements[index], stage.tensor_storage_data_types[index])) {
      invalid_oracle("MatMul argument storage span is invalid");
    }
  }
}

void compute_matmul_host_oracle(const AscendStageArtifact& stage,
                                std::span<const std::uint8_t> a,
                                std::span<const std::uint8_t> b,
                                std::span<std::uint8_t> output) {
  validate_matmul_stage_runtime_contract(stage);
  validate_host_spans(stage, a, b, output);
  for (std::uint64_t logical = 0;
       logical < static_cast<std::uint64_t>(stage.n_elements);
       ++logical) {
    const auto coordinates = output_coordinates(stage, logical);
    const std::uint64_t offset = tensor_offset(
        stage,
        coordinates[0],
        coordinates[1],
        coordinates[2],
        stage.matmul_output_batch_strides,
        stage.matmul_output_stride_m,
        stage.matmul_output_stride_n);
    store_value(output,
                stage.output_type(),
                offset,
                expected_value(stage,
                               a,
                               b,
                               coordinates[0],
                               coordinates[1],
                               coordinates[2]));
  }
}

void seed_matmul_host_inputs(const AscendStageArtifact& stage,
                             std::span<std::uint8_t> a,
                             std::span<std::uint8_t> b) {
  validate_matmul_stage_runtime_contract(stage);
  if (a.size() != stage.arguments[0].size ||
      b.size() != stage.arguments[1].size) {
    invalid_oracle("MatMul seed storage size is invalid");
  }
  std::fill(a.begin(), a.end(), 0xA5U);
  std::fill(b.begin(), b.end(), 0xA5U);
  for (std::uint64_t batch = 0;
       batch < static_cast<std::uint64_t>(stage.matmul_batch);
       ++batch) {
    for (std::uint64_t row = 0;
         row < static_cast<std::uint64_t>(stage.matmul_m);
         ++row) {
      for (std::uint64_t reduction = 0;
           reduction < static_cast<std::uint64_t>(stage.matmul_k);
           ++reduction) {
        const std::uint64_t offset = tensor_offset(
            stage,
            batch,
            row,
            reduction,
            stage.matmul_a_batch_strides,
            stage.matmul_a_stride_m,
            stage.matmul_a_stride_k);
        const std::int64_t code =
            static_cast<std::int64_t>((offset * 17U + 5U) % 29U) - 14;
        store_value(a,
                    stage.input_type(0),
                    offset,
                    static_cast<float>(code) * 0.125F);
      }
    }
    for (std::uint64_t reduction = 0;
         reduction < static_cast<std::uint64_t>(stage.matmul_k);
         ++reduction) {
      for (std::uint64_t column = 0;
           column < static_cast<std::uint64_t>(stage.matmul_n);
           ++column) {
        const std::uint64_t offset = tensor_offset(
            stage,
            batch,
            reduction,
            column,
            stage.matmul_b_batch_strides,
            stage.matmul_b_stride_k,
            stage.matmul_b_stride_n);
        const std::int64_t code =
            static_cast<std::int64_t>((offset * 13U + 11U) % 31U) - 15;
        store_value(b,
                    stage.input_type(1),
                    offset,
                    static_cast<float>(code) * 0.125F);
      }
    }
  }
}

void validate_matmul_host_output(const AscendStageArtifact& stage,
                                 std::span<const std::uint8_t> a,
                                 std::span<const std::uint8_t> b,
                                 std::span<const std::uint8_t> actual) {
  validate_matmul_stage_runtime_contract(stage);
  validate_host_spans(stage, a, b, actual);
  const std::uint64_t logical_count =
      static_cast<std::uint64_t>(stage.n_elements);
  if (logical_count <= kFullOracleOutputLimit) {
    std::vector<std::uint8_t> expected(actual.size(), 0xA5U);
    compute_matmul_host_oracle(stage, a, b, expected);
    std::vector<bool> written(actual.size() /
                                  storage_element_size(stage.output_type()),
                              false);
    for (std::uint64_t logical = 0; logical < logical_count; ++logical) {
      const auto coordinates = output_coordinates(stage, logical);
      const std::uint64_t offset = tensor_offset(
          stage,
          coordinates[0],
          coordinates[1],
          coordinates[2],
          stage.matmul_output_batch_strides,
          stage.matmul_output_stride_m,
          stage.matmul_output_stride_n);
      if (offset >= written.size() || written[offset]) {
        invalid_oracle("MatMul output logical offsets overlap");
      }
      written[offset] = true;
      validate_one_output(stage, a, b, actual, logical);
    }
    const std::size_t size = storage_element_size(stage.output_type());
    for (std::size_t element = 0; element < written.size(); ++element) {
      if (!written[element] &&
          !std::all_of(actual.begin() + element * size,
                       actual.begin() + (element + 1U) * size,
                       [](std::uint8_t byte) { return byte == 0xA5U; })) {
        invalid_oracle("MatMul output padding was modified");
      }
    }
    return;
  }
  const std::uint64_t samples =
      std::min(logical_count, kLargeOracleSampleCount);
  for (std::uint64_t sample = 0; sample < samples; ++sample) {
    const std::uint64_t logical =
        samples == 1U ? 0U : sample * (logical_count - 1U) / (samples - 1U);
    validate_one_output(stage, a, b, actual, logical);
  }
  const std::size_t output_size = storage_element_size(stage.output_type());
  if (actual.size() / output_size != logical_count) {
    std::vector<bool> written(actual.size() / output_size, false);
    for (std::uint64_t logical = 0; logical < logical_count; ++logical) {
      const auto coordinates = output_coordinates(stage, logical);
      const std::uint64_t offset = tensor_offset(
          stage,
          coordinates[0],
          coordinates[1],
          coordinates[2],
          stage.matmul_output_batch_strides,
          stage.matmul_output_stride_m,
          stage.matmul_output_stride_n);
      if (offset >= written.size() || written[offset]) {
        invalid_oracle("MatMul output logical offsets overlap");
      }
      written[offset] = true;
    }
    for (std::size_t element = 0; element < written.size(); ++element) {
      if (!written[element] &&
          !std::all_of(actual.begin() + element * output_size,
                       actual.begin() + (element + 1U) * output_size,
                       [](std::uint8_t byte) { return byte == 0xA5U; })) {
        invalid_oracle("MatMul output padding was modified");
      }
    }
  }
}

void validate_matmul_candidate_outputs(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> reference,
    std::span<const std::uint8_t> candidate) {
  validate_matmul_stage_runtime_contract(stage);
  if (reference.size() != stage.arguments[2].size ||
      candidate.size() != stage.arguments[2].size) {
    invalid_oracle("MatMul candidate output storage size is invalid");
  }
  const std::size_t element_size = storage_element_size(stage.output_type());
  const std::uint64_t logical_count =
      static_cast<std::uint64_t>(stage.n_elements);
  const std::uint64_t samples =
      logical_count <= kFullOracleOutputLimit
          ? logical_count
          : std::min(logical_count, kLargeOracleSampleCount);
  for (std::uint64_t sample = 0; sample < samples; ++sample) {
    const std::uint64_t logical =
        logical_count <= kFullOracleOutputLimit || samples == 1U
            ? sample
            : sample * (logical_count - 1U) / (samples - 1U);
    const auto coordinates = output_coordinates(stage, logical);
    const std::uint64_t offset = tensor_offset(
        stage,
        coordinates[0],
        coordinates[1],
        coordinates[2],
        stage.matmul_output_batch_strides,
        stage.matmul_output_stride_m,
        stage.matmul_output_stride_n);
    const float expected = load_value(reference, stage.output_type(), offset);
    const float observed = load_value(candidate, stage.output_type(), offset);
    const float tolerance = 2.0F * absolute_tolerance(stage) +
                            2.0F * relative_tolerance(stage) *
                                std::max(std::abs(expected), std::abs(observed));
    if (std::abs(observed - expected) > tolerance) {
      invalid_oracle("MatMul autotune candidates differ on logical output");
    }
  }
  if (reference.size() / element_size == logical_count) {
    return;
  }
  std::vector<bool> written(reference.size() / element_size, false);
  for (std::uint64_t logical = 0; logical < logical_count; ++logical) {
    const auto coordinates = output_coordinates(stage, logical);
    const std::uint64_t offset = tensor_offset(
        stage,
        coordinates[0],
        coordinates[1],
        coordinates[2],
        stage.matmul_output_batch_strides,
        stage.matmul_output_stride_m,
        stage.matmul_output_stride_n);
    if (offset >= written.size() || written[offset]) {
      invalid_oracle("MatMul output logical offsets overlap");
    }
    written[offset] = true;
  }
  for (std::size_t element = 0; element < written.size(); ++element) {
    if (written[element]) {
      continue;
    }
    const auto reference_begin = reference.begin() + element * element_size;
    const auto candidate_begin = candidate.begin() + element * element_size;
    if (!std::all_of(reference_begin,
                     reference_begin + element_size,
                     [](std::uint8_t byte) { return byte == 0xA5U; }) ||
        !std::equal(reference_begin,
                    reference_begin + element_size,
                    candidate_begin)) {
      invalid_oracle("MatMul autotune candidate modified output padding");
    }
  }
}

void commit_matmul_host_output(const AscendStageArtifact& stage,
                               std::span<const std::uint8_t> actual,
                               std::span<std::uint8_t> shadow) {
  validate_matmul_stage_runtime_contract(stage);
  if (actual.size() != stage.arguments[2].size ||
      shadow.size() != stage.arguments[2].size) {
    invalid_oracle("MatMul commit storage size is invalid");
  }
  std::copy(actual.begin(), actual.end(), shadow.begin());
}

}  // namespace flagdnn::ascend
