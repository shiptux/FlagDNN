/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/layernorm_oracle.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
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
    return static_cast<std::uint16_t>(sign);
  }
  const std::uint32_t rounded =
      fraction + 0xFFFU + ((fraction >> 13U) & 1U);
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(half_exponent) << 10U) |
      (rounded >> 13U));
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
  stage.operation = "layernorm";
  stage.kernel_family = KernelFamily::kLayerNorm;
  stage.input_count = 3U;
  stage.n_elements = rows * normalized_elements;
  stage.layernorm_rows = rows;
  stage.layernorm_normalized_elements = normalized_elements;
  stage.layernorm_epsilon = 1.0e-5;
  stage.tensor_storage_data_types = {type, type, type, type,
                                     StorageDataType::kFloat32,
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
      pointer(4U, "mean_ptr", 5, statistic_bytes),
      pointer(5U, "inv_variance_ptr", 6, statistic_bytes),
  };
  ArgumentSource scalar;
  scalar.index = 6U;
  scalar.name = "n_elements";
  scalar.source = ArgumentSourceKind::kScalar;
  scalar.type = RawArgumentType::kI32;
  scalar.scalar = stage.n_elements;
  stage.arguments.push_back(scalar);
  return stage;
}

float load(std::span<const std::uint8_t> bytes,
           std::size_t element,
           StorageDataType type) {
  const std::size_t size = element_size(type);
  if (element >= bytes.size() / size) {
    throw std::runtime_error("LayerNorm test storage is too small");
  }
  if (type == StorageDataType::kFloat32) {
    float value = 0.0F;
    std::memcpy(&value, bytes.data() + element * size, sizeof(value));
    return value;
  }
  std::uint16_t encoded = 0U;
  std::memcpy(&encoded, bytes.data() + element * size, sizeof(encoded));
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

void store(std::span<std::uint8_t> bytes,
           std::size_t element,
           StorageDataType type,
           float value) {
  const std::size_t size = element_size(type);
  if (element >= bytes.size() / size) {
    throw std::runtime_error("LayerNorm test storage is too small");
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
  if (flagdnn::ascend::layernorm_kernel_input_count() != 3U ||
      flagdnn::ascend::layernorm_tensor_slot_count() != 6U ||
      flagdnn::ascend::layernorm_runtime_argument_count() != 7U ||
      flagdnn::ascend::layernorm_y_argument_index() != 3U ||
      flagdnn::ascend::layernorm_mean_argument_index() != 4U ||
      flagdnn::ascend::layernorm_inv_variance_argument_index() != 5U) {
    throw std::runtime_error("LayerNorm ABI helper contract mismatch");
  }
  flagdnn::ascend::validate_layernorm_stage_runtime_contract(stage);

  std::vector<std::uint8_t> x(stage.arguments[0].size, 0xA5U);
  std::vector<std::uint8_t> scale(stage.arguments[1].size, 0xA5U);
  std::vector<std::uint8_t> bias(stage.arguments[2].size, 0xA5U);
  flagdnn::ascend::seed_layernorm_host_inputs(stage, x, scale, bias);
  const auto x_snapshot = x;
  const auto scale_snapshot = scale;
  const auto bias_snapshot = bias;
  std::vector<std::uint8_t> second_x(x.size(), 0xA5U);
  std::vector<std::uint8_t> second_scale(scale.size(), 0xA5U);
  std::vector<std::uint8_t> second_bias(bias.size(), 0xA5U);
  flagdnn::ascend::seed_layernorm_host_inputs(
      stage, second_x, second_scale, second_bias);
  if (x != second_x || scale != second_scale || bias != second_bias) {
    throw std::runtime_error("LayerNorm seed is not byte deterministic");
  }

  std::vector<std::uint8_t> y(stage.arguments[3].size, 0xA5U);
  std::vector<std::uint8_t> mean(stage.arguments[4].size, 0xA5U);
  std::vector<std::uint8_t> inv(stage.arguments[5].size, 0xA5U);
  flagdnn::ascend::compute_layernorm_host_oracle(
      stage, x, scale, bias, y, mean, inv);
  flagdnn::ascend::validate_layernorm_host_outputs(
      stage, x, scale, bias, y, mean, inv);
  if (x != x_snapshot || scale != scale_snapshot || bias != bias_snapshot) {
    throw std::runtime_error("LayerNorm host oracle modified an input");
  }

  double sum = 0.0;
  for (std::int32_t column = 0; column < normalized_elements; ++column) {
    sum += load(x, static_cast<std::size_t>(column), type);
  }
  const float expected_mean =
      static_cast<float>(sum / static_cast<double>(normalized_elements));
  double square_sum = 0.0;
  for (std::int32_t column = 0; column < normalized_elements; ++column) {
    const double centered =
        static_cast<double>(load(x, static_cast<std::size_t>(column), type)) -
        expected_mean;
    square_sum += centered * centered;
  }
  const float expected_inv = static_cast<float>(
      1.0 / std::sqrt(square_sum / normalized_elements +
                      stage.layernorm_epsilon));
  if (std::abs(load(mean, 0U, StorageDataType::kFloat32) - expected_mean) >
          2.0e-5F ||
      std::abs(load(inv, 0U, StorageDataType::kFloat32) - expected_inv) >
          2.0e-5F * std::max(1.0F, std::abs(expected_inv))) {
    throw std::runtime_error("LayerNorm statistic formula mismatch");
  }
  const float expected_y =
      (load(x, 0U, type) - expected_mean) * expected_inv *
          load(scale, 0U, type) +
      load(bias, 0U, type);
  const float y_tolerance =
      type == StorageDataType::kFloat32 ? 2.0e-5F : 8.0e-2F;
  if (std::abs(load(y, 0U, type) - expected_y) >
      y_tolerance * std::max(1.0F, std::abs(expected_y))) {
    throw std::runtime_error("LayerNorm Y formula mismatch");
  }

  auto corrupted_y = y;
  store(corrupted_y, 0U, type, load(y, 0U, type) + 64.0F);
  require_invalid(
      [&] {
        flagdnn::ascend::validate_layernorm_host_outputs(
            stage, x, scale, bias, corrupted_y, mean, inv);
      },
      "LayerNorm Y corruption was accepted");
  auto corrupted_mean = mean;
  store(corrupted_mean,
        0U,
        StorageDataType::kFloat32,
        load(mean, 0U, StorageDataType::kFloat32) + 8.0F);
  require_invalid(
      [&] {
        flagdnn::ascend::validate_layernorm_host_outputs(
            stage, x, scale, bias, y, corrupted_mean, inv);
      },
      "LayerNorm mean corruption was accepted");
  auto corrupted_inv = inv;
  store(corrupted_inv,
        0U,
        StorageDataType::kFloat32,
        load(inv, 0U, StorageDataType::kFloat32) + 1.0F);
  require_invalid(
      [&] {
        flagdnn::ascend::validate_layernorm_host_outputs(
            stage, x, scale, bias, y, mean, corrupted_inv);
      },
      "LayerNorm inverse variance corruption was accepted");

  flagdnn::ascend::validate_layernorm_candidate_outputs(
      stage, y, mean, inv, y, mean, inv);
  std::vector<std::uint8_t> shadow_y(y.size(), 0U);
  std::vector<std::uint8_t> shadow_mean(mean.size(), 0U);
  std::vector<std::uint8_t> shadow_inv(inv.size(), 0U);
  flagdnn::ascend::commit_layernorm_host_outputs(
      stage, y, mean, inv, shadow_y, shadow_mean, shadow_inv);
  if (shadow_y != y || shadow_mean != mean || shadow_inv != inv) {
    throw std::runtime_error("LayerNorm full output commit mismatch");
  }

  auto nonfinite = x;
  store(nonfinite, 0U, type, std::numeric_limits<float>::infinity());
  require_invalid(
      [&] {
        flagdnn::ascend::compute_layernorm_host_oracle(
            stage, nonfinite, scale, bias, y, mean, inv);
      },
      "LayerNorm nonfinite X was accepted");

}

void check_contract_rejections() {
  AscendStageArtifact stage =
      make_stage(StorageDataType::kFloat32, 10, 17);
  stage.arguments[6].type = RawArgumentType::kI64;
  require_invalid(
      [&] { flagdnn::ascend::validate_layernorm_stage_runtime_contract(stage); },
      "LayerNorm scalar ABI mutation was accepted");

  stage = make_stage(StorageDataType::kFloat32, 10, 17);
  stage.arguments[5].uid = stage.arguments[4].uid;
  require_invalid(
      [&] { flagdnn::ascend::validate_layernorm_stage_runtime_contract(stage); },
      "LayerNorm duplicate output UID was accepted");

  stage = make_stage(StorageDataType::kFloat32, 10, 17);
  stage.tensor_storage_data_types[4] = StorageDataType::kFloat16;
  require_invalid(
      [&] { flagdnn::ascend::validate_layernorm_stage_runtime_contract(stage); },
      "LayerNorm non-FP32 statistic type was accepted");

  stage = make_stage(StorageDataType::kFloat32, 10, 17);
  stage.arguments[1].size -= sizeof(float);
  require_invalid(
      [&] { flagdnn::ascend::validate_layernorm_stage_runtime_contract(stage); },
      "LayerNorm short parameter metadata was accepted");

  stage = make_stage(StorageDataType::kFloat32, 10, 17);
  std::vector<std::uint8_t> shared(stage.arguments[0].size, 0U);
  std::vector<std::uint8_t> scale(stage.arguments[1].size, 0U);
  std::vector<std::uint8_t> bias(stage.arguments[2].size, 0U);
  std::vector<std::uint8_t> mean(stage.arguments[4].size, 0U);
  std::vector<std::uint8_t> inv(stage.arguments[5].size, 0U);
  flagdnn::ascend::seed_layernorm_host_inputs(stage, shared, scale, bias);
  require_invalid(
      [&] {
        flagdnn::ascend::compute_layernorm_host_oracle(
            stage, shared, scale, bias, shared, mean, inv);
      },
      "LayerNorm overlapping X/Y storage was accepted");

  std::vector<std::uint8_t> y(stage.arguments[3].size, 0xA5U);
  std::vector<std::uint8_t> short_shadow(y.size() - 1U, 0U);
  require_invalid(
      [&] {
        flagdnn::ascend::commit_layernorm_host_outputs(
            stage, y, mean, inv, short_shadow, mean, inv);
      },
      "LayerNorm short commit span was accepted");

  stage = make_stage(StorageDataType::kFloat16, 8, 4096);
  for (std::size_t index : {3U, 4U, 5U}) {
    stage.arguments[index].source = ArgumentSourceKind::kGraphWorkspace;
    stage.arguments[index].alignment = 256U;
    stage.arguments[index].workspace_offset = index * 4096U;
  }
  flagdnn::ascend::validate_layernorm_stage_runtime_contract(stage);
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
