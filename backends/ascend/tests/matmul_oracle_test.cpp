/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/matmul_oracle.hpp"

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
#include <span>
#include <stdexcept>
#include <string>
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

AscendStageArtifact make_stage(StorageDataType type, bool broadcast = false) {
  AscendStageArtifact stage;
  stage.operation = "matmul";
  stage.kernel_family = KernelFamily::kMatMul;
  stage.tensor_storage_data_types = {type, type, type};
  stage.input_count = 2U;
  stage.matmul_batch = broadcast ? 6 : 2;
  stage.matmul_m = 3;
  stage.matmul_n = 4;
  stage.matmul_k = 5;
  stage.n_elements = static_cast<std::int32_t>(
      stage.matmul_batch * stage.matmul_m * stage.matmul_n);
  stage.matmul_batch_dimensions =
      broadcast ? std::array<std::int64_t, 6>{1, 1, 1, 1, 2, 3}
                : std::array<std::int64_t, 6>{1, 1, 1, 1, 1, 2};
  stage.matmul_a_batch_strides =
      broadcast ? std::array<std::int64_t, 6>{0, 0, 0, 0, 32, 0}
                : std::array<std::int64_t, 6>{0, 0, 0, 0, 0, 32};
  stage.matmul_b_batch_strides =
      broadcast ? std::array<std::int64_t, 6>{0, 0, 0, 0, 0, 40}
                : std::array<std::int64_t, 6>{0, 0, 0, 0, 0, 40};
  stage.matmul_output_batch_strides =
      broadcast ? std::array<std::int64_t, 6>{0, 0, 0, 0, 72, 24}
                : std::array<std::int64_t, 6>{0, 0, 0, 0, 0, 24};
  stage.matmul_a_stride_m = 7;
  stage.matmul_a_stride_k = 1;
  stage.matmul_b_stride_k = 6;
  stage.matmul_b_stride_n = 1;
  stage.matmul_output_stride_m = 6;
  stage.matmul_output_stride_n = 1;
  const std::size_t size = element_size(type);
  const std::size_t a_elements = 51U;
  const std::size_t b_elements = broadcast ? 108U : 68U;
  const std::size_t output_elements = broadcast ? 136U : 40U;
  stage.arguments = {
      pointer_argument(0U, "a_ptr", 1, a_elements * size),
      pointer_argument(1U, "b_ptr", 2, b_elements * size),
      pointer_argument(2U, "output_ptr", 3, output_elements * size),
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

void check_case(StorageDataType type, bool broadcast) {
  AscendStageArtifact stage = make_stage(type, broadcast);
  flagdnn::ascend::validate_matmul_stage_runtime_contract(stage);
  const std::size_t size = element_size(type);
  std::vector<std::uint8_t> a(stage.arguments[0].size, 0xA5U);
  std::vector<std::uint8_t> b(stage.arguments[1].size, 0xA5U);
  flagdnn::ascend::seed_matmul_host_inputs(stage, a, b);
  const std::vector<std::uint8_t> first_a = a;
  const std::vector<std::uint8_t> first_b = b;
  std::fill(a.begin(), a.end(), 0xA5U);
  std::fill(b.begin(), b.end(), 0xA5U);
  flagdnn::ascend::seed_matmul_host_inputs(stage, a, b);
  if (a != first_a || b != first_b) {
    throw std::runtime_error("MatMul seed is not byte reproducible");
  }
  std::vector<std::uint8_t> output(stage.arguments[2].size, 0xA5U);
  flagdnn::ascend::compute_matmul_host_oracle(stage, a, b, output);
  if (a != first_a || b != first_b) {
    throw std::runtime_error("MatMul host oracle changed an input");
  }
  double manual = 0.0;
  for (std::size_t k = 0; k < 5U; ++k) {
    manual += static_cast<double>(load_value(a, k, type)) *
              static_cast<double>(load_value(b, k * 6U, type));
  }
  const float observed = load_value(output, 0U, type);
  const float tolerance = type == StorageDataType::kFloat32 ? 1.0e-5F : 0.2F;
  if (std::abs(observed - static_cast<float>(manual)) >
      tolerance * std::max(1.0F, std::abs(static_cast<float>(manual)))) {
    throw std::runtime_error("MatMul host oracle differs from manual FP32 sum");
  }
  flagdnn::ascend::validate_matmul_host_output(stage, a, b, output);
  std::vector<std::uint8_t> candidate = output;
  const float candidate_value = load_value(candidate, 0U, type);
  store_value(candidate,
              0U,
              type,
              candidate_value +
                  (type == StorageDataType::kFloat32 ? 1.0e-6F : 1.0e-3F));
  flagdnn::ascend::validate_matmul_candidate_outputs(
      stage, output, candidate);
  if (!std::all_of(output.begin() + 4U * size,
                   output.begin() + 5U * size,
                   [](std::uint8_t byte) { return byte == 0xA5U; })) {
    throw std::runtime_error("MatMul host oracle overwrote output padding");
  }
  std::vector<std::uint8_t> corrupted = output;
  store_value(corrupted,
              0U,
              type,
              load_value(corrupted, 0U, type) + 64.0F);
  expect_invalid(
      [&] {
        flagdnn::ascend::validate_matmul_host_output(stage, a, b, corrupted);
      },
      "MatMul logical output corruption was accepted");
  corrupted = output;
  corrupted[4U * size] ^= 1U;
  expect_invalid(
      [&] {
        flagdnn::ascend::validate_matmul_host_output(stage, a, b, corrupted);
      },
      "MatMul output padding corruption was accepted");
  expect_invalid(
      [&] {
        flagdnn::ascend::validate_matmul_candidate_outputs(
            stage, output, corrupted);
      },
      "MatMul candidate padding corruption was accepted");
  std::vector<std::uint8_t> shadow(output.size(), 0U);
  flagdnn::ascend::commit_matmul_host_output(stage, output, shadow);
  if (shadow != output) {
    throw std::runtime_error("MatMul full output shadow was not committed");
  }
  expect_invalid(
      [&] {
        flagdnn::ascend::commit_matmul_host_output(
            stage, output, std::span<std::uint8_t>(shadow).first(shadow.size() - 1U));
      },
      "MatMul short commit shadow was accepted");
}

void check_runtime_and_rejections() {
  if (flagdnn::ascend::matmul_kernel_input_count() != 2U ||
      flagdnn::ascend::matmul_output_argument_index() != 2U ||
      flagdnn::ascend::matmul_tensor_slot_count() != 3U ||
      flagdnn::ascend::matmul_runtime_argument_count() != 4U ||
      std::string(flagdnn::ascend::matmul_output_argument_name()) !=
          "output_ptr") {
    throw std::runtime_error("MatMul runtime routing constants are invalid");
  }
  AscendStageArtifact stage = make_stage(StorageDataType::kFloat32);
  flagdnn::ascend::validate_matmul_stage_runtime_contract(stage);
  AscendStageArtifact changed = stage;
  changed.arguments[1].name = "rhs_ptr";
  expect_invalid(
      [&] { flagdnn::ascend::validate_matmul_stage_runtime_contract(changed); },
      "MatMul ABI name mutation was accepted");
  changed = stage;
  changed.arguments[3].type = RawArgumentType::kI64;
  expect_invalid(
      [&] { flagdnn::ascend::validate_matmul_stage_runtime_contract(changed); },
      "MatMul scalar ABI type mutation was accepted");
  changed = stage;
  changed.matmul_output_stride_m = 1;
  expect_invalid(
      [&] { flagdnn::ascend::validate_matmul_stage_runtime_contract(changed); },
      "MatMul overlapping output strides were accepted");
  changed = stage;
  changed.matmul_batch = 3;
  expect_invalid(
      [&] { flagdnn::ascend::validate_matmul_stage_runtime_contract(changed); },
      "MatMul inconsistent batch metadata was accepted");

  std::vector<std::uint8_t> a(stage.arguments[0].size, 0xA5U);
  std::vector<std::uint8_t> b(stage.arguments[1].size, 0xA5U);
  std::vector<std::uint8_t> output(stage.arguments[2].size, 0xA5U);
  flagdnn::ascend::seed_matmul_host_inputs(stage, a, b);
  store_value(a, 0U, StorageDataType::kFloat32,
              std::numeric_limits<float>::infinity());
  expect_invalid(
      [&] { flagdnn::ascend::compute_matmul_host_oracle(stage, a, b, output); },
      "MatMul nonfinite input was accepted");
  flagdnn::ascend::seed_matmul_host_inputs(stage, a, b);
  flagdnn::ascend::compute_matmul_host_oracle(stage, a, b, output);
  store_value(output, 0U, StorageDataType::kFloat32,
              std::numeric_limits<float>::infinity());
  expect_invalid(
      [&] {
        flagdnn::ascend::validate_matmul_host_output(stage, a, b, output);
      },
      "MatMul nonfinite output was accepted");
  expect_invalid(
      [&] {
        flagdnn::ascend::compute_matmul_host_oracle(
            stage,
            a,
            b,
            std::span<std::uint8_t>(a).first(stage.arguments[2].size));
      },
      "MatMul output/input memory overlap was accepted");
  expect_invalid(
      [&] {
        flagdnn::ascend::compute_matmul_host_oracle(
            stage,
            std::span<const std::uint8_t>(a).first(a.size() - 1U),
            b,
            output);
      },
      "MatMul short A storage was accepted");
}

}  // namespace

int main() {
  try {
    check_case(StorageDataType::kFloat32, false);
    check_case(StorageDataType::kFloat16, false);
    check_case(StorageDataType::kBFloat16, false);
    check_case(StorageDataType::kBFloat16, true);
    check_runtime_and_rejections();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
