/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/batchnorm_oracle.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

using namespace flagdnn::ascend;

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

std::size_t element_size(StorageDataType type) {
  return type == StorageDataType::kFloat32 ? 4U : 2U;
}

std::size_t required_elements(
    const std::array<std::int64_t, 8>& dimensions,
    const std::array<std::int64_t, 8>& strides) {
  std::size_t maximum = 0U;
  for (std::size_t axis = 0U; axis < dimensions.size(); ++axis) {
    maximum += static_cast<std::size_t>(dimensions[axis] - 1) *
               static_cast<std::size_t>(strides[axis]);
  }
  return maximum + 1U;
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

AscendStageArtifact make_stage(
    StorageDataType type = StorageDataType::kFloat32,
    bool gapped = false) {
  AscendStageArtifact stage;
  stage.operation = "batchnorm";
  stage.kernel_family = KernelFamily::kBatchNorm;
  stage.input_count = 5U;
  stage.n_elements = 24;
  stage.batchnorm_rank = 4;
  stage.batchnorm_batch = 2;
  stage.batchnorm_channels = 3;
  stage.batchnorm_spatial = 4;
  stage.batchnorm_reduction_elements = 8;
  stage.batchnorm_epsilon = 1.0e-3;
  stage.batchnorm_momentum = 0.1;
  stage.batchnorm_dimensions = {1, 1, 1, 1, 2, 3, 2, 2};
  stage.batchnorm_x_strides =
      gapped ? std::array<std::int64_t, 8>{0, 0, 0, 0, 30, 9, 3, 1}
             : std::array<std::int64_t, 8>{0, 0, 0, 0, 12, 4, 2, 1};
  stage.batchnorm_y_strides =
      gapped ? std::array<std::int64_t, 8>{0, 0, 0, 0, 40, 12, 4, 1}
             : stage.batchnorm_x_strides;
  stage.tensor_storage_data_types = {
      type, type, type, StorageDataType::kFloat32,
      StorageDataType::kFloat32, type,
      StorageDataType::kFloat32, StorageDataType::kFloat32,
      StorageDataType::kFloat32, StorageDataType::kFloat32};
  static constexpr std::array<const char*, 10> names = {
      "x_ptr", "scale_ptr", "bias_ptr", "previous_running_mean_ptr",
      "previous_running_variance_ptr", "y_ptr", "mean_ptr",
      "inv_variance_ptr", "next_running_mean_ptr",
      "next_running_variance_ptr"};
  const std::size_t x_bytes =
      required_elements(stage.batchnorm_dimensions,
                        stage.batchnorm_x_strides) * element_size(type);
  const std::size_t y_bytes =
      required_elements(stage.batchnorm_dimensions,
                        stage.batchnorm_y_strides) * element_size(type);
  const std::size_t value_bytes = 3U * element_size(type);
  for (std::size_t index = 0; index < names.size(); ++index) {
    const std::size_t bytes = index == 0U ? x_bytes
                               : index == 5U ? y_bytes
                               : index == 1U || index == 2U ? value_bytes
                                                            : 12U;
    stage.arguments.push_back(
        pointer(index, names[index], static_cast<std::int64_t>(index + 1),
                bytes));
  }
  ArgumentSource scalar;
  scalar.index = 10U;
  scalar.name = "n_elements";
  scalar.source = ArgumentSourceKind::kScalar;
  scalar.type = RawArgumentType::kI32;
  scalar.scalar = stage.n_elements;
  stage.arguments.push_back(scalar);
  return stage;
}

float load(std::span<const std::uint8_t> bytes,
           std::size_t index,
           StorageDataType type = StorageDataType::kFloat32) {
  const std::size_t size = element_size(type);
  if (type == StorageDataType::kFloat32) {
    float value = 0.0F;
    std::memcpy(&value, bytes.data() + index * size, sizeof(value));
    return value;
  }
  std::uint16_t encoded = 0U;
  std::memcpy(&encoded, bytes.data() + index * size, sizeof(encoded));
  if (type == StorageDataType::kBFloat16) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(encoded) << 16U);
  }
  const std::uint32_t sign = static_cast<std::uint32_t>(encoded & 0x8000U)
                             << 16U;
  const std::uint32_t exponent = (encoded >> 10U) & 0x1FU;
  const std::uint32_t fraction = encoded & 0x3FFU;
  return exponent == 0U
             ? std::bit_cast<float>(sign)
             : std::bit_cast<float>(sign | ((exponent + 112U) << 23U) |
                                    (fraction << 13U));
}

void store(std::span<std::uint8_t> bytes,
           std::size_t index,
           float value,
           StorageDataType type = StorageDataType::kFloat32) {
  const std::size_t size = element_size(type);
  if (type == StorageDataType::kFloat32) {
    std::memcpy(bytes.data() + index * size, &value, sizeof(value));
    return;
  }
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint16_t encoded =
      type == StorageDataType::kBFloat16
          ? static_cast<std::uint16_t>(
                (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U)
          : float_to_half(value);
  std::memcpy(bytes.data() + index * size, &encoded, size);
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

void run_contract(StorageDataType type, bool gapped) {
  AscendStageArtifact stage = make_stage(type, gapped);
  if (batchnorm_kernel_input_count() != 5U ||
      batchnorm_tensor_slot_count() != 10U ||
      batchnorm_runtime_argument_count() != 11U ||
      batchnorm_first_output_argument_index() != 5U) {
    throw std::runtime_error("BatchNorm ABI helpers are invalid");
  }
  validate_batchnorm_stage_runtime_contract(stage);

  std::array<std::vector<std::uint8_t>, 5> input_storage;
  for (std::size_t index = 0U; index < input_storage.size(); ++index) {
    input_storage[index].assign(stage.arguments[index].size, 0xA5U);
  }
  BatchNormHostBuffers inputs = {
      input_storage[0], input_storage[1], input_storage[2], input_storage[3],
      input_storage[4]};
  seed_batchnorm_host_inputs(stage, inputs);
  const auto snapshots = input_storage;
  auto second_seed = input_storage;
  for (auto& storage : second_seed) {
    std::fill(storage.begin(), storage.end(), 0xA5U);
  }
  BatchNormHostBuffers second_inputs = {
      second_seed[0], second_seed[1], second_seed[2], second_seed[3],
      second_seed[4]};
  seed_batchnorm_host_inputs(stage, second_inputs);
  if (second_seed != input_storage) {
    throw std::runtime_error("BatchNorm seed is not byte deterministic");
  }

  std::array<std::vector<std::uint8_t>, 5> output_storage;
  for (std::size_t index = 0U; index < output_storage.size(); ++index) {
    output_storage[index].assign(stage.arguments[index + 5U].size, 0xA5U);
  }
  BatchNormHostBuffers outputs = {
      output_storage[0], output_storage[1], output_storage[2],
      output_storage[3], output_storage[4]};
  ConstBatchNormHostBuffers const_inputs = {
      input_storage[0], input_storage[1], input_storage[2], input_storage[3],
      input_storage[4]};
  compute_batchnorm_host_oracle(stage, const_inputs, outputs);
  ConstBatchNormHostBuffers const_outputs = {
      output_storage[0], output_storage[1], output_storage[2],
      output_storage[3], output_storage[4]};
  validate_batchnorm_host_outputs(stage, const_inputs, const_outputs);
  if (input_storage != snapshots) {
    throw std::runtime_error("BatchNorm host oracle modified an input");
  }

  const std::array<std::size_t, 8> channel_zero_offsets =
      gapped ? std::array<std::size_t, 8>{0U, 1U, 3U, 4U, 30U, 31U, 33U, 34U}
             : std::array<std::size_t, 8>{0U, 1U, 2U, 3U, 12U, 13U, 14U, 15U};
  double sum = 0.0;
  for (std::size_t index : channel_zero_offsets) {
    sum += load(input_storage[0], index, type);
  }
  const double mean = sum / 8.0;
  double square_sum = 0.0;
  for (std::size_t index : channel_zero_offsets) {
    const double centered = load(input_storage[0], index, type) - mean;
    square_sum += centered * centered;
  }
  const double variance = square_sum / 8.0;
  const double inv = 1.0 / std::sqrt(variance + stage.batchnorm_epsilon);
  const double unbiased = variance * 8.0 / 7.0;
  if (std::abs(load(output_storage[1], 0U) - mean) > 2.0e-5 ||
      std::abs(load(output_storage[2], 0U) - inv) > 2.0e-5 ||
      std::abs(load(output_storage[3], 0U) -
               (load(input_storage[3], 0U) * 0.9 + mean * 0.1)) > 2.0e-5 ||
      std::abs(load(output_storage[4], 0U) -
               (load(input_storage[4], 0U) * 0.9 + unbiased * 0.1)) >
          2.0e-5) {
    throw std::runtime_error("BatchNorm training statistic formula mismatch");
  }

  validate_batchnorm_candidate_outputs(stage, const_outputs, const_outputs);
  auto corrupted = output_storage;
  store(corrupted[1], 0U, load(corrupted[1], 0U) + 4.0F);
  ConstBatchNormHostBuffers corrupted_outputs = {
      corrupted[0], corrupted[1], corrupted[2], corrupted[3], corrupted[4]};
  require_invalid(
      [&] { validate_batchnorm_host_outputs(stage, const_inputs,
                                            corrupted_outputs); },
      "BatchNorm corrupted mean was accepted");

  std::array<std::vector<std::uint8_t>, 5> shadow_storage;
  for (std::size_t index = 0U; index < shadow_storage.size(); ++index) {
    shadow_storage[index].resize(stage.arguments[index + 5U].size);
  }
  BatchNormHostBuffers shadows = {
      shadow_storage[0], shadow_storage[1], shadow_storage[2],
      shadow_storage[3], shadow_storage[4]};
  commit_batchnorm_host_outputs(stage, const_outputs, shadows);
  if (shadow_storage != output_storage) {
    throw std::runtime_error("BatchNorm full output commit mismatch");
  }

  auto corrupted_y = output_storage;
  store(corrupted_y[0], 0U, load(corrupted_y[0], 0U, type) + 64.0F, type);
  ConstBatchNormHostBuffers corrupted_y_outputs = {
      corrupted_y[0], corrupted_y[1], corrupted_y[2], corrupted_y[3],
      corrupted_y[4]};
  require_invalid(
      [&] { validate_batchnorm_host_outputs(stage, const_inputs,
                                            corrupted_y_outputs); },
      "BatchNorm corrupted Y was accepted");

  if (gapped) {
    auto corrupted_padding = output_storage;
    const std::size_t y_elements = corrupted_padding[0].size() / element_size(type);
    std::vector<bool> logical(y_elements, false);
    for (std::size_t n = 0U; n < 2U; ++n) {
      for (std::size_t c = 0U; c < 3U; ++c) {
        for (std::size_t h = 0U; h < 2U; ++h) {
          for (std::size_t w = 0U; w < 2U; ++w) {
            logical[n * 40U + c * 12U + h * 4U + w] = true;
          }
        }
      }
    }
    const auto padding = std::find(logical.begin(), logical.end(), false);
    if (padding == logical.end()) {
      throw std::runtime_error("BatchNorm gapped fixture has no padding");
    }
    corrupted_padding[0][static_cast<std::size_t>(padding - logical.begin()) *
                            element_size(type)] = 0U;
    ConstBatchNormHostBuffers corrupted_padding_outputs = {
        corrupted_padding[0], corrupted_padding[1], corrupted_padding[2],
        corrupted_padding[3], corrupted_padding[4]};
    require_invalid(
        [&] { validate_batchnorm_host_outputs(stage, const_inputs,
                                              corrupted_padding_outputs); },
        "BatchNorm Y padding corruption was accepted");
  }

  auto short_shadow_storage = shadow_storage;
  short_shadow_storage[4].pop_back();
  BatchNormHostBuffers short_shadows = {
      short_shadow_storage[0], short_shadow_storage[1], short_shadow_storage[2],
      short_shadow_storage[3], short_shadow_storage[4]};
  require_invalid(
      [&] { commit_batchnorm_host_outputs(stage, const_outputs, short_shadows); },
      "BatchNorm short commit span was accepted");

  BatchNormHostBuffers overlapping_outputs = {
      input_storage[0], output_storage[1], output_storage[2], output_storage[3],
      output_storage[4]};
  require_invalid(
      [&] { compute_batchnorm_host_oracle(stage, const_inputs,
                                          overlapping_outputs); },
      "BatchNorm overlapping X/Y storage was accepted");

  stage.arguments[10].type = RawArgumentType::kI64;
  require_invalid(
      [&] { validate_batchnorm_stage_runtime_contract(stage); },
      "BatchNorm scalar ABI mutation was accepted");
}

}  // namespace

int main() {
  try {
    run_contract(StorageDataType::kFloat32, false);
    run_contract(StorageDataType::kFloat16, true);
    run_contract(StorageDataType::kBFloat16, true);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  return 0;
}
