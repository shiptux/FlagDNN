/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/rmsnorm_oracle.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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
  return type == StorageDataType::kFloat32 ? 4U : 2U;
}

std::uint16_t float_to_half(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t absolute = bits & 0x7FFFFFFFU;
  if (absolute >= 0x7F800000U) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (absolute < 0x38800000U) {
    return static_cast<std::uint16_t>(sign);
  }
  const std::uint32_t rounded = absolute + 0x00001000U;
  return static_cast<std::uint16_t>(
      sign | ((rounded - 0x38000000U) >> 13U));
}

void store(std::span<std::uint8_t> bytes,
           std::size_t element,
           StorageDataType type,
           float value) {
  const std::size_t size = element_size(type);
  if (element >= bytes.size() / size) {
    throw std::runtime_error("test storage is too small");
  }
  if (type == StorageDataType::kFloat32) {
    std::memcpy(bytes.data() + element * size, &value, size);
    return;
  }
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint16_t encoded =
      type == StorageDataType::kBFloat16
          ? static_cast<std::uint16_t>(
                (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U)
          : float_to_half(value);
  std::memcpy(bytes.data() + element * size, &encoded, size);
}

float load(std::span<const std::uint8_t> bytes,
           std::size_t element,
           StorageDataType type) {
  const std::size_t size = element_size(type);
  if (element >= bytes.size() / size) {
    throw std::runtime_error("test storage is too small");
  }
  if (type == StorageDataType::kFloat32) {
    float result = 0.0F;
    std::memcpy(&result, bytes.data() + element * size, size);
    return result;
  }
  std::uint16_t encoded = 0;
  std::memcpy(&encoded, bytes.data() + element * size, size);
  if (type == StorageDataType::kBFloat16) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(encoded) << 16U);
  }
  const std::uint32_t sign = static_cast<std::uint32_t>(encoded & 0x8000U)
                             << 16U;
  const std::uint32_t exponent = (encoded >> 10U) & 0x1FU;
  const std::uint32_t fraction = encoded & 0x3FFU;
  if (exponent == 0U) {
    return std::bit_cast<float>(sign);
  }
  return std::bit_cast<float>(
      sign | ((exponent + 112U) << 23U) | (fraction << 13U));
}

ArgumentSource pointer(std::size_t index,
                       const char* name,
                       std::int64_t uid,
                       std::size_t size) {
  ArgumentSource result;
  result.index = index;
  result.name = name;
  result.source = ArgumentSourceKind::kBinding;
  result.type = RawArgumentType::kPointer;
  result.uid = uid;
  result.size = size;
  result.alignment = 16U;
  return result;
}

AscendStageArtifact make_stage(StorageDataType type,
                               std::int32_t rows,
                               std::int32_t normalized_elements) {
  AscendStageArtifact stage;
  stage.operation = "rmsnorm";
  stage.kernel_family = KernelFamily::kRmsNorm;
  stage.input_count = 3U;
  stage.n_elements = rows * normalized_elements;
  stage.rmsnorm_rows = rows;
  stage.rmsnorm_normalized_elements = normalized_elements;
  stage.rmsnorm_epsilon = 1.0e-5;
  stage.tensor_storage_data_types = {type, type, type, type,
                                     StorageDataType::kFloat32};
  const std::size_t value_size = element_size(type);
  const std::size_t xy_bytes =
      static_cast<std::size_t>(stage.n_elements) * value_size;
  const std::size_t parameter_bytes =
      static_cast<std::size_t>(normalized_elements) * value_size;
  const std::size_t statistic_bytes =
      static_cast<std::size_t>(rows) * sizeof(float);
  stage.arguments = {
      pointer(0U, "x_ptr", 1, xy_bytes),
      pointer(1U, "scale_ptr", 2, parameter_bytes),
      pointer(2U, "bias_ptr", 3, parameter_bytes),
      pointer(3U, "y_ptr", 4, xy_bytes),
      pointer(4U, "inv_variance_ptr", 5, statistic_bytes),
  };
  ArgumentSource scalar;
  scalar.index = 5U;
  scalar.name = "n_elements";
  scalar.source = ArgumentSourceKind::kScalar;
  scalar.type = RawArgumentType::kI32;
  scalar.scalar = stage.n_elements;
  stage.arguments.push_back(scalar);
  return stage;
}

template <typename Function>
void require_invalid(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

void check_case(StorageDataType type,
                std::int32_t rows,
                std::int32_t normalized_elements) {
  AscendStageArtifact stage = make_stage(type, rows, normalized_elements);
  flagdnn::ascend::validate_rmsnorm_stage_runtime_contract(stage);
  const std::size_t value_size = element_size(type);
  std::vector<std::uint8_t> x(stage.arguments[0].size, 0xA5U);
  std::vector<std::uint8_t> scale(stage.arguments[1].size, 0xA5U);
  std::vector<std::uint8_t> bias(stage.arguments[2].size, 0xA5U);
  flagdnn::ascend::seed_rmsnorm_host_inputs(stage, x, scale, bias);
  const auto x_snapshot = x;
  const auto scale_snapshot = scale;
  const auto bias_snapshot = bias;
  std::vector<std::uint8_t> second_x(x.size(), 0xA5U);
  std::vector<std::uint8_t> second_scale(scale.size(), 0xA5U);
  std::vector<std::uint8_t> second_bias(bias.size(), 0xA5U);
  flagdnn::ascend::seed_rmsnorm_host_inputs(
      stage, second_x, second_scale, second_bias);
  if (second_x != x || second_scale != scale || second_bias != bias) {
    throw std::runtime_error("RMSNorm seed is not byte deterministic");
  }

  std::vector<std::uint8_t> y(stage.arguments[3].size, 0xA5U);
  std::vector<std::uint8_t> inv(stage.arguments[4].size, 0xA5U);
  flagdnn::ascend::compute_rmsnorm_host_oracle(
      stage, x, scale, bias, y, inv);
  flagdnn::ascend::validate_rmsnorm_host_outputs(
      stage, x, scale, bias, y, inv);
  if (x != x_snapshot || scale != scale_snapshot || bias != bias_snapshot) {
    throw std::runtime_error("RMSNorm host oracle modified an input");
  }

  double square_sum = 0.0;
  for (std::int32_t index = 0; index < normalized_elements; ++index) {
    const float value = load(x, static_cast<std::size_t>(index), type);
    square_sum += static_cast<double>(value) * value;
  }
  const float expected_inv = static_cast<float>(
      1.0 / std::sqrt(square_sum / normalized_elements + stage.rmsnorm_epsilon));
  const float observed_inv = load(inv, 0U, StorageDataType::kFloat32);
  if (std::abs(observed_inv - expected_inv) >
      2.0e-5F * std::max(1.0F, std::abs(expected_inv))) {
    throw std::runtime_error("RMSNorm inverse variance formula mismatch");
  }
  const float expected_y =
      load(x, 0U, type) * expected_inv * load(scale, 0U, type) +
      load(bias, 0U, type);
  const float observed_y = load(y, 0U, type);
  const float y_tolerance =
      type == StorageDataType::kFloat32 ? 2.0e-5F : 8.0e-2F;
  if (std::abs(observed_y - expected_y) >
      y_tolerance * std::max(1.0F, std::abs(expected_y))) {
    throw std::runtime_error("RMSNorm Y formula mismatch");
  }

  auto corrupted_y = y;
  store(corrupted_y, 0U, type, observed_y + 64.0F);
  require_invalid(
      [&] {
        flagdnn::ascend::validate_rmsnorm_host_outputs(
            stage, x, scale, bias, corrupted_y, inv);
      },
      "RMSNorm Y corruption was accepted");
  auto corrupted_inv = inv;
  store(corrupted_inv, 0U, StorageDataType::kFloat32, observed_inv + 1.0F);
  require_invalid(
      [&] {
        flagdnn::ascend::validate_rmsnorm_host_outputs(
            stage, x, scale, bias, y, corrupted_inv);
      },
      "RMSNorm inverse variance corruption was accepted");

  flagdnn::ascend::validate_rmsnorm_candidate_outputs(
      stage, y, inv, y, inv);
  std::vector<std::uint8_t> shadow_y(y.size(), 0U);
  std::vector<std::uint8_t> shadow_inv(inv.size(), 0U);
  flagdnn::ascend::commit_rmsnorm_host_outputs(
      stage, y, inv, shadow_y, shadow_inv);
  if (shadow_y != y || shadow_inv != inv) {
    throw std::runtime_error("RMSNorm full output commit mismatch");
  }

  auto nonfinite = x;
  store(nonfinite, 0U, type, std::numeric_limits<float>::infinity());
  require_invalid(
      [&] {
        flagdnn::ascend::compute_rmsnorm_host_oracle(
            stage, nonfinite, scale, bias, y, inv);
      },
      "RMSNorm nonfinite X was accepted");
  (void)value_size;
}

void check_contract_rejections() {
  if (flagdnn::ascend::rmsnorm_kernel_input_count() != 3U ||
      flagdnn::ascend::rmsnorm_tensor_slot_count() != 5U ||
      flagdnn::ascend::rmsnorm_runtime_argument_count() != 6U ||
      flagdnn::ascend::rmsnorm_y_argument_index() != 3U ||
      flagdnn::ascend::rmsnorm_inv_variance_argument_index() != 4U) {
    throw std::runtime_error("RMSNorm ABI helper contract mismatch");
  }
  AscendStageArtifact stage =
      make_stage(StorageDataType::kFloat32, 10, 17);
  stage.arguments[5].type = RawArgumentType::kI64;
  require_invalid(
      [&] { flagdnn::ascend::validate_rmsnorm_stage_runtime_contract(stage); },
      "RMSNorm scalar ABI mutation was accepted");
  stage = make_stage(StorageDataType::kFloat32, 10, 17);
  stage.arguments[4].uid = stage.arguments[3].uid;
  require_invalid(
      [&] { flagdnn::ascend::validate_rmsnorm_stage_runtime_contract(stage); },
      "RMSNorm duplicate output UID was accepted");

  stage = make_stage(StorageDataType::kFloat32, 10, 17);
  std::vector<std::uint8_t> shared(stage.arguments[0].size, 0U);
  std::vector<std::uint8_t> scale(stage.arguments[1].size, 0U);
  std::vector<std::uint8_t> bias(stage.arguments[2].size, 0U);
  std::vector<std::uint8_t> inv(stage.arguments[4].size, 0U);
  flagdnn::ascend::seed_rmsnorm_host_inputs(stage, shared, scale, bias);
  require_invalid(
      [&] {
        flagdnn::ascend::compute_rmsnorm_host_oracle(
            stage, shared, scale, bias, shared, inv);
      },
      "RMSNorm overlapping X/Y storage was accepted");

  std::vector<std::uint8_t> y(stage.arguments[3].size, 0xA5U);
  std::vector<std::uint8_t> short_shadow(y.size() - 1U, 0U);
  require_invalid(
      [&] {
        flagdnn::ascend::commit_rmsnorm_host_outputs(
            stage, y, inv, short_shadow, inv);
      },
      "RMSNorm short commit span was accepted");
}

}  // namespace

int main() {
  try {
    check_case(StorageDataType::kFloat32, 10, 17);
    check_case(StorageDataType::kFloat16, 8, 4096);
    check_case(StorageDataType::kBFloat16, 6, 20);
    check_contract_rejections();
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
