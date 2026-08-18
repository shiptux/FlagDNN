/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/batchnorm_inference_oracle.hpp"
#include "backends/ascend/engines/build_time_prewarm.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>
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
  const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
  const std::uint32_t fraction = bits & 0x7FFFFFU;
  if (exponent == 0xFFU) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  const std::int32_t half_exponent = static_cast<std::int32_t>(exponent) - 112;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (half_exponent <= 0) {
    return static_cast<std::uint16_t>(sign);
  }
  const std::uint32_t rounded = fraction + 0xFFFU + ((fraction >> 13U) & 1U);
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(half_exponent) << 10U) |
      (rounded >> 13U));
}

float half_to_float(std::uint16_t encoded) {
  const std::uint32_t sign = static_cast<std::uint32_t>(encoded & 0x8000U)
                             << 16U;
  const std::uint32_t exponent = (encoded >> 10U) & 0x1FU;
  const std::uint32_t fraction = encoded & 0x3FFU;
  if (exponent == 0U) {
    return std::bit_cast<float>(sign);
  }
  return std::bit_cast<float>(sign | ((exponent + 112U) << 23U) |
                              (fraction << 13U));
}

void store_value(std::vector<std::uint8_t>& storage,
                 std::size_t element,
                 StorageDataType type,
                 float value) {
  const std::size_t size = element_size(type);
  if (type == StorageDataType::kFloat32) {
    std::memcpy(storage.data() + element * size, &value, size);
    return;
  }
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint16_t encoded =
      type == StorageDataType::kBFloat16
          ? static_cast<std::uint16_t>(
                (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U)
          : float_to_half(value);
  std::memcpy(storage.data() + element * size, &encoded, size);
}

float load_value(const std::vector<std::uint8_t>& storage,
                 std::size_t element,
                 StorageDataType type) {
  const std::size_t size = element_size(type);
  if (type == StorageDataType::kFloat32) {
    float result = 0.0F;
    std::memcpy(&result, storage.data() + element * size, size);
    return result;
  }
  std::uint16_t encoded = 0;
  std::memcpy(&encoded, storage.data() + element * size, size);
  if (type == StorageDataType::kBFloat16) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(encoded) << 16U);
  }
  return half_to_float(encoded);
}

std::array<std::int64_t, 8> padded(const std::vector<std::int64_t>& values,
                                   std::int64_t leading) {
  std::array<std::int64_t, 8> result{};
  result.fill(leading);
  std::copy(values.begin(), values.end(), result.end() - values.size());
  return result;
}

std::size_t storage_elements(const std::vector<std::int64_t>& dimensions,
                             const std::vector<std::int64_t>& strides) {
  std::size_t maximum = 0;
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    maximum += static_cast<std::size_t>(dimensions[axis] - 1) *
               static_cast<std::size_t>(strides[axis]);
  }
  return maximum + 1U;
}

std::size_t tensor_offset(std::size_t logical,
                          const std::vector<std::int64_t>& dimensions,
                          const std::vector<std::int64_t>& strides) {
  std::size_t result = 0;
  for (std::size_t reversed = dimensions.size(); reversed != 0; --reversed) {
    const std::size_t axis = reversed - 1U;
    const std::size_t coordinate =
        logical % static_cast<std::size_t>(dimensions[axis]);
    logical /= static_cast<std::size_t>(dimensions[axis]);
    result += coordinate * static_cast<std::size_t>(strides[axis]);
  }
  return result;
}

ArgumentSource pointer_argument(std::size_t index,
                                const char* name,
                                bool is_virtual,
                                std::int64_t uid,
                                std::size_t offset) {
  ArgumentSource result;
  result.index = index;
  result.name = name;
  result.source = is_virtual ? ArgumentSourceKind::kGraphWorkspace
                             : ArgumentSourceKind::kBinding;
  result.type = RawArgumentType::kPointer;
  result.uid = uid;
  result.size = 64;
  result.alignment = is_virtual ? 256 : 16;
  result.workspace_offset = is_virtual ? offset : 0;
  return result;
}

AscendStageArtifact make_stage(StorageDataType type,
                               const std::vector<std::int64_t>& dimensions,
                               const std::vector<std::int64_t>& x_strides,
                               const std::vector<std::int64_t>& y_strides,
                               bool virtual_xy = false) {
  AscendStageArtifact stage;
  stage.operation = "batchnorm_inference";
  stage.kernel_family = KernelFamily::kBatchNormInference;
  stage.tensor_storage_data_types = {
      type, StorageDataType::kFloat32, StorageDataType::kFloat32,
      StorageDataType::kFloat32, StorageDataType::kFloat32, type};
  stage.input_count = 5;
  stage.n_elements = 1;
  for (const std::int64_t dimension : dimensions) {
    stage.n_elements *= static_cast<std::int32_t>(dimension);
  }
  stage.batchnorm_rank = static_cast<std::int64_t>(dimensions.size());
  stage.batchnorm_channels = dimensions[1];
  stage.batchnorm_spatial = 1;
  for (std::size_t axis = 2; axis < dimensions.size(); ++axis) {
    stage.batchnorm_spatial *= dimensions[axis];
  }
  stage.batchnorm_dimensions = padded(dimensions, 1);
  stage.batchnorm_x_strides = padded(x_strides, 0);
  stage.batchnorm_y_strides = padded(y_strides, 0);
  stage.arguments = {
      pointer_argument(0, "x_ptr", virtual_xy, 1, 0),
      pointer_argument(1, "mean_ptr", false, 2, 0),
      pointer_argument(2, "inv_variance_ptr", false, 3, 0),
      pointer_argument(3, "scale_ptr", false, 4, 0),
      pointer_argument(4, "bias_ptr", false, 5, 0),
      pointer_argument(5, "y_ptr", virtual_xy, 6, 256),
  };
  stage.arguments[0].size =
      storage_elements(dimensions, x_strides) * element_size(type);
  for (std::size_t index = 1; index < 5U; ++index) {
    stage.arguments[index].size =
        static_cast<std::size_t>(stage.batchnorm_channels) * sizeof(float);
  }
  stage.arguments[5].size =
      storage_elements(dimensions, y_strides) * element_size(type);
  ArgumentSource scalar;
  scalar.index = 6;
  scalar.name = "n_elements";
  scalar.source = ArgumentSourceKind::kScalar;
  scalar.type = RawArgumentType::kI32;
  scalar.scalar = stage.n_elements;
  stage.arguments.push_back(scalar);
  return stage;
}

template <typename Function>
void expect_invalid(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

void check_case(StorageDataType type,
                const std::vector<std::int64_t>& dimensions,
                const std::vector<std::int64_t>& x_strides,
                const std::vector<std::int64_t>& y_strides,
                bool virtual_xy) {
  AscendStageArtifact stage =
      make_stage(type, dimensions, x_strides, y_strides, virtual_xy);
  flagdnn::ascend::validate_batchnorm_inference_stage_runtime_contract(stage);
  const std::size_t x_elements = storage_elements(dimensions, x_strides);
  const std::size_t y_elements = storage_elements(dimensions, y_strides);
  const std::size_t channels = static_cast<std::size_t>(dimensions[1]);
  std::vector<std::uint8_t> x(x_elements * element_size(type), 0xA5U);
  std::vector<std::uint8_t> mean(channels * sizeof(float), 0U);
  std::vector<std::uint8_t> inv_variance(channels * sizeof(float), 0U);
  std::vector<std::uint8_t> scale(channels * sizeof(float), 0U);
  std::vector<std::uint8_t> bias(channels * sizeof(float), 0U);
  std::vector<std::uint8_t> y(y_elements * element_size(type), 0xA5U);
  const std::array<float, 8> mean_codes = {
      32.0F, -32.0F, 0.0001F, -0.0001F, 4.0F, -4.0F, 16.0F, -16.0F};
  const std::array<float, 8> inv_codes = {
      1.0F / 64.0F, 8.0F, 0.125F, 2.0F, 0.5F, 4.0F, 0.25F, 1.0F};
  const std::array<float, 8> scale_codes = {
      -8.0F, 4.0F, 0.5F, -2.0F, 1.25F, -0.75F, 3.0F, -4.0F};
  const std::array<float, 8> bias_codes = {
      100.0F, -200.0F, 0.001F, -0.001F, 32.0F, -64.0F, 8.0F, -16.0F};
  std::vector<float> means(channels);
  std::vector<float> invs(channels);
  std::vector<float> scales(channels);
  std::vector<float> biases(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const std::size_t code = channel % mean_codes.size();
    means[channel] = mean_codes[code];
    invs[channel] = inv_codes[code];
    scales[channel] = scale_codes[code];
    biases[channel] = bias_codes[code];
    store_value(mean, channel, StorageDataType::kFloat32, means[channel]);
    store_value(inv_variance,
                channel,
                StorageDataType::kFloat32,
                invs[channel]);
    store_value(scale, channel, StorageDataType::kFloat32, scales[channel]);
    store_value(bias, channel, StorageDataType::kFloat32, biases[channel]);
  }
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const float value = static_cast<float>((logical * 7) % 17) - 8.0F;
    store_value(x,
                tensor_offset(static_cast<std::size_t>(logical),
                              dimensions,
                              x_strides),
                type,
                value);
  }
  const auto x_before = x;
  const auto mean_before = mean;
  const auto inv_before = inv_variance;
  const auto scale_before = scale;
  const auto bias_before = bias;
  flagdnn::ascend::compute_batchnorm_inference_host_oracle(
      stage, x, mean, inv_variance, scale, bias, y);
  if (x != x_before || mean != mean_before || inv_variance != inv_before ||
      scale != scale_before || bias != bias_before) {
    throw std::runtime_error("batchnorm oracle modified an input tensor");
  }
  std::vector<bool> x_written(x_elements, false);
  std::vector<bool> written(y_elements, false);
  const std::size_t spatial = static_cast<std::size_t>(stage.batchnorm_spatial);
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const std::size_t index = static_cast<std::size_t>(logical);
    const std::size_t channel = (index / spatial) % channels;
    const std::size_t x_offset = tensor_offset(index, dimensions, x_strides);
    const std::size_t y_offset = tensor_offset(index, dimensions, y_strides);
    x_written[x_offset] = true;
    written[y_offset] = true;
    const float expected =
        (load_value(x, x_offset, type) - means[channel]) * invs[channel] *
            scales[channel] +
        biases[channel];
    const float actual = load_value(y, y_offset, type);
    const float tolerance =
        type == StorageDataType::kFloat32 ? 1.0e-5F : 2.0e-2F;
    if (std::abs(actual - expected) >
        tolerance * std::max(1.0F, std::abs(expected))) {
      throw std::runtime_error("batchnorm host oracle value mismatch");
    }
  }
  const std::size_t size = element_size(type);
  for (std::size_t element = 0; element < x_written.size(); ++element) {
    if (!x_written[element] &&
        !std::all_of(x.begin() + element * size,
                     x.begin() + (element + 1U) * size,
                     [](std::uint8_t byte) { return byte == 0xA5U; })) {
      throw std::runtime_error("batchnorm oracle modified X padding");
    }
  }
  for (std::size_t element = 0; element < written.size(); ++element) {
    if (!written[element] &&
        !std::all_of(y.begin() + element * size,
                     y.begin() + (element + 1U) * size,
                     [](std::uint8_t byte) { return byte == 0xA5U; })) {
      throw std::runtime_error("batchnorm host oracle overwrote Y padding");
    }
  }
  flagdnn::ascend::validate_batchnorm_inference_host_output(
      stage, x, mean, inv_variance, scale, bias, y);
  std::vector<std::uint8_t> corrupted = y;
  const std::size_t first_y = tensor_offset(0, dimensions, y_strides);
  const float first_value = load_value(corrupted, first_y, type);
  store_value(corrupted,
              first_y,
              type,
              first_value + std::max(64.0F, std::abs(first_value) * 2.0F));
  expect_invalid(
      [&]() {
        flagdnn::ascend::validate_batchnorm_inference_host_output(
            stage, x, mean, inv_variance, scale, bias, corrupted);
      },
      "batchnorm logical Y corruption was accepted");
  const auto padding = std::find(written.begin(), written.end(), false);
  if (padding != written.end()) {
    corrupted = y;
    corrupted[static_cast<std::size_t>(padding - written.begin()) * size] = 0U;
    expect_invalid(
        [&]() {
          flagdnn::ascend::validate_batchnorm_inference_host_output(
              stage, x, mean, inv_variance, scale, bias, corrupted);
        },
        "batchnorm Y padding corruption was accepted");
  }
  std::vector<std::uint8_t> shadow(y.size(), 0U);
  flagdnn::ascend::commit_batchnorm_inference_host_output(stage, y, shadow);
  if (shadow != y) {
    throw std::runtime_error("batchnorm virtual shadow commit mismatch");
  }
}

void check_candidate_output_comparison(StorageDataType type) {
  const std::vector<std::int64_t> dimensions = {2, 3, 2};
  const std::vector<std::int64_t> strides = {20, 5, 2};
  const AscendStageArtifact stage =
      make_stage(type, dimensions, strides, strides, false);
  const std::size_t elements = storage_elements(dimensions, strides);
  const std::size_t size = element_size(type);
  std::vector<std::uint8_t> reference(elements * size, 0xA5U);
  std::vector<bool> written(elements, false);
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const std::size_t offset = tensor_offset(
        static_cast<std::size_t>(logical), dimensions, strides);
    written[offset] = true;
    store_value(reference,
                offset,
                type,
                static_cast<float>(logical) * 0.25F - 1.0F);
  }
  std::vector<std::uint8_t> candidate = reference;
  flagdnn::ascend::validate_batchnorm_inference_candidate_outputs(
      stage, reference, candidate);

  const std::size_t first = tensor_offset(0, dimensions, strides);
  store_value(candidate,
              first,
              type,
              load_value(candidate, first, type) + 64.0F);
  expect_invalid(
      [&]() {
        flagdnn::ascend::validate_batchnorm_inference_candidate_outputs(
            stage, reference, candidate);
      },
      "batchnorm cross-candidate logical corruption was accepted");

  const auto padding = std::find(written.begin(), written.end(), false);
  if (padding == written.end()) {
    throw std::runtime_error("candidate comparator fixture has no padding");
  }
  candidate = reference;
  candidate[static_cast<std::size_t>(padding - written.begin()) * size] = 0U;
  expect_invalid(
      [&]() {
        flagdnn::ascend::validate_batchnorm_inference_candidate_outputs(
            stage, reference, candidate);
      },
      "batchnorm cross-candidate padding corruption was accepted");
}

void check_runtime_contract_and_seeding() {
  if (flagdnn::ascend::batchnorm_inference_kernel_input_count() != 5U ||
      flagdnn::ascend::batchnorm_inference_tensor_slot_count() != 6U ||
      flagdnn::ascend::batchnorm_inference_output_argument_index() != 5U ||
      flagdnn::ascend::batchnorm_inference_runtime_argument_count() != 7U ||
      std::string(
          flagdnn::ascend::batchnorm_inference_output_argument_name()) !=
          "y_ptr") {
    throw std::runtime_error("batchnorm engine ABI routing mismatch");
  }
  AscendStageArtifact stage = make_stage(StorageDataType::kFloat32,
                                         {2, 3},
                                         {3, 1},
                                         {4, 1},
                                         true);
  std::vector<std::uint8_t> x(6U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> mean(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> inv_variance(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> scale(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> bias(3U * sizeof(float), 0xA5U);
  flagdnn::ascend::seed_batchnorm_inference_host_inputs(
      stage, x, mean, inv_variance, scale, bias);
  for (const auto* buffer : {&x, &mean, &inv_variance, &scale, &bias}) {
    for (std::size_t element = 0; element < buffer->size() / sizeof(float);
         ++element) {
      if (!std::isfinite(load_value(
              *buffer, element, StorageDataType::kFloat32))) {
        throw std::runtime_error("batchnorm seed is not finite");
      }
    }
  }
  std::vector<float> x_values;
  for (std::size_t logical = 0; logical < 6U; ++logical) {
    x_values.push_back(
        load_value(x, tensor_offset(logical, {2, 3}, {3, 1}),
                   StorageDataType::kFloat32));
  }
  std::sort(x_values.begin(), x_values.end());
  if (std::adjacent_find(x_values.begin(), x_values.end()) != x_values.end() ||
      std::all_of(x_values.begin(), x_values.end(),
                  [](float value) { return value == 0.0F; })) {
    throw std::runtime_error("batchnorm seed does not distinguish X positions");
  }
  const std::array<float, 3> seeded_means = {
      load_value(mean, 0, StorageDataType::kFloat32),
      load_value(mean, 1, StorageDataType::kFloat32),
      load_value(mean, 2, StorageDataType::kFloat32)};
  const std::array<float, 3> seeded_invs = {
      load_value(inv_variance, 0, StorageDataType::kFloat32),
      load_value(inv_variance, 1, StorageDataType::kFloat32),
      load_value(inv_variance, 2, StorageDataType::kFloat32)};
  const std::array<float, 3> seeded_scales = {
      load_value(scale, 0, StorageDataType::kFloat32),
      load_value(scale, 1, StorageDataType::kFloat32),
      load_value(scale, 2, StorageDataType::kFloat32)};
  const std::array<float, 3> seeded_biases = {
      load_value(bias, 0, StorageDataType::kFloat32),
      load_value(bias, 1, StorageDataType::kFloat32),
      load_value(bias, 2, StorageDataType::kFloat32)};
  const auto has_signs = [](const auto& values) {
    return std::any_of(values.begin(), values.end(),
                       [](float value) { return value < 0.0F; }) &&
           std::any_of(values.begin(), values.end(),
                       [](float value) { return value > 0.0F; });
  };
  if (!has_signs(seeded_means) || !has_signs(seeded_scales) ||
      !has_signs(seeded_biases) ||
      std::max({std::abs(seeded_means[0]), std::abs(seeded_means[1]),
                std::abs(seeded_means[2])}) < 16.0F ||
      *std::min_element(seeded_invs.begin(), seeded_invs.end()) > 1.0F / 64.0F ||
      *std::max_element(seeded_invs.begin(), seeded_invs.end()) < 8.0F ||
      std::max({std::abs(seeded_scales[0]), std::abs(seeded_scales[1]),
                std::abs(seeded_scales[2])}) < 4.0F ||
      std::max({std::abs(seeded_biases[0]), std::abs(seeded_biases[1]),
                std::abs(seeded_biases[2])}) < 100.0F) {
    throw std::runtime_error("batchnorm seed misses finite extreme parameters");
  }
  store_value(mean,
              0,
              StorageDataType::kFloat32,
              std::numeric_limits<float>::infinity());
  std::vector<std::uint8_t> y(7U * sizeof(float), 0xA5U);
  try {
    flagdnn::ascend::compute_batchnorm_inference_host_oracle(
        stage, x, mean, inv_variance, scale, bias, y);
    throw std::runtime_error("nonfinite batchnorm parameter was accepted");
  } catch (const std::invalid_argument&) {
  }
}

void check_nonfinite_inputs_and_actual() {
  AscendStageArtifact stage = make_stage(StorageDataType::kFloat32,
                                         {2, 3},
                                         {3, 1},
                                         {4, 1},
                                         false);
  std::vector<std::uint8_t> x(6U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> mean(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> inv_variance(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> scale(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> bias(3U * sizeof(float), 0xA5U);
  flagdnn::ascend::seed_batchnorm_inference_host_inputs(
      stage, x, mean, inv_variance, scale, bias);
  const auto reject_nonfinite = [&](std::size_t slot, float value) {
    auto changed_x = x;
    auto changed_mean = mean;
    auto changed_inv = inv_variance;
    auto changed_scale = scale;
    auto changed_bias = bias;
    std::array<std::vector<std::uint8_t>*, 5> inputs = {
        &changed_x, &changed_mean, &changed_inv, &changed_scale, &changed_bias};
    store_value(*inputs[slot], 0, StorageDataType::kFloat32, value);
    std::vector<std::uint8_t> y(7U * sizeof(float), 0xA5U);
    expect_invalid(
        [&]() {
          flagdnn::ascend::compute_batchnorm_inference_host_oracle(
              stage,
              changed_x,
              changed_mean,
              changed_inv,
              changed_scale,
              changed_bias,
              y);
        },
        "nonfinite batchnorm input was accepted");
  };
  for (std::size_t slot = 0; slot < 5U; ++slot) {
    reject_nonfinite(slot, std::numeric_limits<float>::quiet_NaN());
    reject_nonfinite(slot, std::numeric_limits<float>::infinity());
  }
  std::vector<std::uint8_t> y(7U * sizeof(float), 0xA5U);
  flagdnn::ascend::compute_batchnorm_inference_host_oracle(
      stage, x, mean, inv_variance, scale, bias, y);
  store_value(y,
              0,
              StorageDataType::kFloat32,
              std::numeric_limits<float>::quiet_NaN());
  expect_invalid(
      [&]() {
        flagdnn::ascend::validate_batchnorm_inference_host_output(
            stage, x, mean, inv_variance, scale, bias, y);
      },
      "nonfinite batchnorm actual output was accepted");
  store_value(y,
              0,
              StorageDataType::kFloat32,
              std::numeric_limits<float>::infinity());
  expect_invalid(
      [&]() {
        flagdnn::ascend::validate_batchnorm_inference_host_output(
            stage, x, mean, inv_variance, scale, bias, y);
      },
      "infinite batchnorm actual output was accepted");
}

void check_low_precision_nonfinite(StorageDataType type) {
  AscendStageArtifact stage =
      make_stage(type, {1, 3}, {3, 1}, {3, 1}, false);
  std::vector<std::uint8_t> x(3U * element_size(type), 0xA5U);
  std::vector<std::uint8_t> mean(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> inv_variance(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> scale(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> bias(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> y(3U * element_size(type), 0xA5U);
  flagdnn::ascend::seed_batchnorm_inference_host_inputs(
      stage, x, mean, inv_variance, scale, bias);
  flagdnn::ascend::compute_batchnorm_inference_host_oracle(
      stage, x, mean, inv_variance, scale, bias, y);

  const std::uint16_t infinity =
      type == StorageDataType::kFloat16 ? 0x7C00U : 0x7F80U;
  std::memcpy(y.data(), &infinity, sizeof(infinity));
  expect_invalid(
      [&]() {
        flagdnn::ascend::validate_batchnorm_inference_host_output(
            stage, x, mean, inv_variance, scale, bias, y);
      },
      "low-precision infinite batchnorm actual output was accepted");

  std::memcpy(x.data(), &infinity, sizeof(infinity));
  std::fill(y.begin(), y.end(), 0xA5U);
  expect_invalid(
      [&]() {
        flagdnn::ascend::compute_batchnorm_inference_host_oracle(
            stage, x, mean, inv_variance, scale, bias, y);
      },
      "low-precision infinite batchnorm X was accepted");
}

void check_fractional_rounding(StorageDataType type) {
  AscendStageArtifact stage =
      make_stage(type, {1, 3}, {3, 1}, {4, 1}, false);
  std::vector<std::uint8_t> x(3U * element_size(type), 0xA5U);
  std::vector<std::uint8_t> mean(3U * sizeof(float), 0U);
  std::vector<std::uint8_t> inv_variance(3U * sizeof(float), 0U);
  std::vector<std::uint8_t> scale(3U * sizeof(float), 0U);
  std::vector<std::uint8_t> bias(3U * sizeof(float), 0U);
  std::vector<std::uint8_t> y(3U * element_size(type), 0xA5U);
  for (std::size_t channel = 0; channel < 3U; ++channel) {
    store_value(x, channel, type, 0.333251953125F + channel * 0.125F);
    store_value(mean,
                channel,
                StorageDataType::kFloat32,
                -0.111328125F + channel * 0.03125F);
    store_value(inv_variance,
                channel,
                StorageDataType::kFloat32,
                0.66650390625F + channel * 0.0625F);
    store_value(scale,
                channel,
                StorageDataType::kFloat32,
                channel == 1U ? -1.375F : 1.1875F);
    store_value(bias,
                channel,
                StorageDataType::kFloat32,
                0.0078125F * static_cast<float>(channel + 1U));
  }
  flagdnn::ascend::compute_batchnorm_inference_host_oracle(
      stage, x, mean, inv_variance, scale, bias, y);
  std::vector<std::uint8_t> encoded_expected(y.size(), 0xA5U);
  for (std::size_t channel = 0; channel < 3U; ++channel) {
    const float expected =
        (load_value(x, channel, type) -
         load_value(mean, channel, StorageDataType::kFloat32)) *
            load_value(inv_variance, channel, StorageDataType::kFloat32) *
            load_value(scale, channel, StorageDataType::kFloat32) +
        load_value(bias, channel, StorageDataType::kFloat32);
    store_value(encoded_expected, channel, type, expected);
  }
  if (y != encoded_expected) {
    throw std::runtime_error("batchnorm fractional storage rounding mismatch");
  }
}

void check_alias_and_overlap_rejection() {
  AscendStageArtifact x_alias_stage = make_stage(StorageDataType::kFloat32,
                                                 {2, 3},
                                                 {0, 0},
                                                 {3, 1},
                                                 false);
  expect_invalid(
      [&]() {
        flagdnn::ascend::validate_batchnorm_inference_stage_runtime_contract(
            x_alias_stage);
      },
      "batchnorm X stride alias was accepted");
  std::vector<std::uint8_t> x(sizeof(float), 0xA5U);
  std::vector<std::uint8_t> mean(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> inv_variance(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> scale(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> bias(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> y(6U * sizeof(float), 0xA5U);

  AscendStageArtifact y_alias_stage = make_stage(StorageDataType::kFloat32,
                                                 {2, 3},
                                                 {3, 1},
                                                 {0, 0},
                                                 false);
  expect_invalid(
      [&]() {
        flagdnn::ascend::validate_batchnorm_inference_stage_runtime_contract(
            y_alias_stage);
      },
      "batchnorm Y stride alias was accepted");

  AscendStageArtifact stage = make_stage(StorageDataType::kFloat32,
                                         {2, 3},
                                         {3, 1},
                                         {3, 1},
                                         false);
  x.assign(6U * sizeof(float), 0xA5U);
  mean.assign(3U * sizeof(float), 0xA5U);
  inv_variance.assign(3U * sizeof(float), 0xA5U);
  scale.assign(3U * sizeof(float), 0xA5U);
  bias.assign(3U * sizeof(float), 0xA5U);
  flagdnn::ascend::seed_batchnorm_inference_host_inputs(
      stage, x, mean, inv_variance, scale, bias);
  for (std::size_t slot = 0; slot < 5U; ++slot) {
    auto changed_x = x;
    auto changed_mean = mean;
    auto changed_inv = inv_variance;
    auto changed_scale = scale;
    auto changed_bias = bias;
    std::vector<std::uint8_t> alias(6U * sizeof(float), 0xA5U);
    std::array<std::vector<std::uint8_t>*, 5> inputs = {
        &changed_x, &changed_mean, &changed_inv, &changed_scale, &changed_bias};
    const std::size_t input_bytes = slot == 0U ? alias.size()
                                                : 3U * sizeof(float);
    std::copy(inputs[slot]->begin(), inputs[slot]->end(), alias.begin());
    std::span<const std::uint8_t> changed_x_span = changed_x;
    std::span<const std::uint8_t> changed_mean_span = changed_mean;
    std::span<const std::uint8_t> changed_inv_span = changed_inv;
    std::span<const std::uint8_t> changed_scale_span = changed_scale;
    std::span<const std::uint8_t> changed_bias_span = changed_bias;
    const std::span<const std::uint8_t> alias_input(alias.data(), input_bytes);
    switch (slot) {
      case 0:
        changed_x_span = alias_input;
        break;
      case 1:
        changed_mean_span = alias_input;
        break;
      case 2:
        changed_inv_span = alias_input;
        break;
      case 3:
        changed_scale_span = alias_input;
        break;
      case 4:
        changed_bias_span = alias_input;
        break;
    }
    expect_invalid(
        [&]() {
          flagdnn::ascend::compute_batchnorm_inference_host_oracle(
              stage,
              changed_x_span,
              changed_mean_span,
              changed_inv_span,
              changed_scale_span,
              changed_bias_span,
              alias);
        },
        "batchnorm Y/input storage overlap was accepted");
  }

  for (std::size_t channel = 0; channel < 3U; ++channel) {
    store_value(mean, channel, StorageDataType::kFloat32, 0.0F);
    store_value(inv_variance, channel, StorageDataType::kFloat32, 1.0F);
    store_value(scale, channel, StorageDataType::kFloat32, 1.0F);
    store_value(bias, channel, StorageDataType::kFloat32, 0.0F);
  }
  expect_invalid(
      [&]() {
        flagdnn::ascend::validate_batchnorm_inference_host_output(
            stage, x, mean, inv_variance, scale, bias, x);
      },
      "batchnorm actual/X storage overlap was accepted");
}

void check_short_storage_rejection() {
  AscendStageArtifact stage = make_stage(StorageDataType::kFloat32,
                                         {2, 3},
                                         {3, 1},
                                         {4, 1},
                                         false);
  std::vector<std::uint8_t> x(6U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> mean(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> inv_variance(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> scale(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> bias(3U * sizeof(float), 0xA5U);
  flagdnn::ascend::seed_batchnorm_inference_host_inputs(
      stage, x, mean, inv_variance, scale, bias);
  std::vector<std::uint8_t> y(7U * sizeof(float), 0xA5U);
  const auto reject_sizes = [&](std::size_t slot) {
    auto short_x = x;
    auto short_mean = mean;
    auto short_inv = inv_variance;
    auto short_scale = scale;
    auto short_bias = bias;
    auto short_y = y;
    std::array<std::vector<std::uint8_t>*, 6> buffers = {
        &short_x, &short_mean, &short_inv, &short_scale, &short_bias, &short_y};
    buffers[slot]->resize(buffers[slot]->size() - sizeof(float));
    expect_invalid(
        [&]() {
          flagdnn::ascend::compute_batchnorm_inference_host_oracle(
              stage,
              short_x,
              short_mean,
              short_inv,
              short_scale,
              short_bias,
              short_y);
        },
        "short batchnorm storage was accepted");
  };
  for (std::size_t slot = 0; slot < 6U; ++slot) {
    reject_sizes(slot);
  }
}

void check_rank8_and_malformed_metadata() {
  const std::vector<std::int64_t> dimensions = {1, 3, 1, 1, 1, 1, 1, 2};
  const std::vector<std::int64_t> x_strides = {6, 2, 2, 2, 2, 2, 2, 1};
  const std::vector<std::int64_t> y_strides = {12, 4, 4, 4, 4, 4, 4, 2};
  check_case(StorageDataType::kFloat32,
             dimensions,
             x_strides,
             y_strides,
             false);
  const AscendStageArtifact baseline = make_stage(StorageDataType::kFloat32,
                                                  dimensions,
                                                  x_strides,
                                                  y_strides,
                                                  false);
  const auto reject = [&baseline](auto mutate, const char* message) {
    AscendStageArtifact changed = baseline;
    mutate(changed);
    expect_invalid(
        [&]() {
          flagdnn::ascend::validate_batchnorm_inference_stage_runtime_contract(
              changed);
        },
        message);
  };
  reject([](AscendStageArtifact& stage) {
           stage.batchnorm_dimensions[7] = 3;
         },
         "batchnorm dimension/n_elements mismatch was accepted");
  reject([](AscendStageArtifact& stage) { stage.batchnorm_channels = 2; },
         "batchnorm channel decomposition mismatch was accepted");
  reject([](AscendStageArtifact& stage) { stage.batchnorm_spatial = 1; },
         "batchnorm spatial decomposition mismatch was accepted");
  reject([](AscendStageArtifact& stage) {
           stage.batchnorm_x_strides[7] = -1;
         },
         "negative batchnorm X stride was accepted");
  reject([](AscendStageArtifact& stage) {
           stage.batchnorm_y_strides[7] = -1;
         },
         "negative batchnorm Y stride was accepted");
}

void check_commit_and_source_metadata() {
  AscendStageArtifact virtual_stage = make_stage(StorageDataType::kFloat32,
                                                 {2, 3},
                                                 {3, 1},
                                                 {4, 1},
                                                 true);
  std::vector<std::uint8_t> x(6U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> mean(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> inv_variance(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> scale(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> bias(3U * sizeof(float), 0xA5U);
  flagdnn::ascend::seed_batchnorm_inference_host_inputs(
      virtual_stage, x, mean, inv_variance, scale, bias);
  std::vector<std::uint8_t> actual(7U * sizeof(float), 0xA5U);
  flagdnn::ascend::compute_batchnorm_inference_host_oracle(
      virtual_stage, x, mean, inv_variance, scale, bias, actual);
  std::vector<std::uint8_t> shadow(actual.size(), 0U);
  flagdnn::ascend::commit_batchnorm_inference_host_output(
      virtual_stage, actual, shadow);
  if (shadow != actual || shadow[3U * sizeof(float)] != 0xA5U) {
    throw std::runtime_error("batchnorm commit missed full Y storage padding");
  }

  auto short_actual = actual;
  short_actual.resize(short_actual.size() - sizeof(float));
  std::vector<std::uint8_t> short_shadow(short_actual.size(), 0U);
  expect_invalid(
      [&]() {
        flagdnn::ascend::commit_batchnorm_inference_host_output(
            virtual_stage, short_actual, short_shadow);
      },
      "equally short batchnorm commit spans were accepted");
  auto long_actual = actual;
  long_actual.resize(long_actual.size() + sizeof(float), 0xA5U);
  std::vector<std::uint8_t> long_shadow(long_actual.size(), 0U);
  expect_invalid(
      [&]() {
        flagdnn::ascend::commit_batchnorm_inference_host_output(
            virtual_stage, long_actual, long_shadow);
      },
      "equally long batchnorm commit spans were accepted");
  shadow.resize(actual.size() - sizeof(float));
  expect_invalid(
      [&]() {
        flagdnn::ascend::commit_batchnorm_inference_host_output(
            virtual_stage, actual, shadow);
      },
      "mismatched batchnorm commit spans were accepted");

  const auto reject_stage = [&virtual_stage](auto mutate,
                                              const char* message) {
    AscendStageArtifact changed = virtual_stage;
    mutate(changed);
    expect_invalid(
        [&]() {
          flagdnn::ascend::validate_batchnorm_inference_stage_runtime_contract(
              changed);
        },
        message);
  };
  reject_stage([](AscendStageArtifact& stage) {
                 stage.arguments[5].size += sizeof(float);
               },
               "wrong batchnorm Y argument size was accepted");
  reject_stage([](AscendStageArtifact& stage) {
                 stage.arguments[0].size -= sizeof(float);
               },
               "wrong batchnorm X argument size was accepted");
  reject_stage([](AscendStageArtifact& stage) {
                 stage.arguments[2].size += sizeof(float);
               },
               "wrong batchnorm parameter argument size was accepted");
  reject_stage([](AscendStageArtifact& stage) {
                 stage.arguments[5].workspace_offset = 1;
               },
               "misaligned virtual batchnorm Y offset was accepted");
  reject_stage([](AscendStageArtifact& stage) {
                 stage.arguments[0].alignment = 16;
               },
               "weak virtual batchnorm X alignment was accepted");

  AscendStageArtifact binding_stage = make_stage(StorageDataType::kFloat32,
                                                 {2, 3},
                                                 {3, 1},
                                                 {3, 1},
                                                 false);
  binding_stage.arguments[5].workspace_offset = 256;
  expect_invalid(
      [&]() {
        flagdnn::ascend::validate_batchnorm_inference_stage_runtime_contract(
            binding_stage);
      },
      "binding batchnorm Y carried a workspace offset");
}

void check_negative_stage_contract() {
  const AscendStageArtifact baseline = make_stage(StorageDataType::kFloat32,
                                                  {2, 3},
                                                  {3, 1},
                                                  {3, 1},
                                                  false);
  const auto reject = [&baseline](auto mutate, const char* message) {
    AscendStageArtifact changed = baseline;
    mutate(changed);
    expect_invalid(
        [&changed]() {
          flagdnn::ascend::validate_batchnorm_inference_stage_runtime_contract(
              changed);
        },
        message);
  };
  reject([](AscendStageArtifact& stage) {
           stage.kernel_family = KernelFamily::kBinary;
         },
         "wrong batchnorm family was accepted");
  reject([](AscendStageArtifact& stage) { stage.operation = "add"; },
         "wrong batchnorm operation was accepted");
  reject([](AscendStageArtifact& stage) { stage.input_count = 4; },
         "wrong batchnorm input count was accepted");
  reject([](AscendStageArtifact& stage) {
           stage.tensor_storage_data_types.pop_back();
         },
         "wrong batchnorm tensor slot count was accepted");
  reject([](AscendStageArtifact& stage) {
           stage.tensor_storage_data_types[2] = StorageDataType::kFloat16;
         },
         "non-float32 batchnorm parameter was accepted");
  reject([](AscendStageArtifact& stage) { stage.arguments[0].index = 1; },
         "wrong batchnorm ABI index was accepted");
  reject([](AscendStageArtifact& stage) {
           stage.arguments[2].name = "variance_ptr";
         },
         "wrong batchnorm ABI name was accepted");
  reject([](AscendStageArtifact& stage) {
           stage.arguments[3].source = ArgumentSourceKind::kGraphWorkspace;
         },
         "virtual batchnorm parameter was accepted");
  reject([](AscendStageArtifact& stage) {
           stage.arguments[6].source = ArgumentSourceKind::kBinding;
         },
         "non-scalar batchnorm n_elements was accepted");
  reject([](AscendStageArtifact& stage) {
           stage.arguments[6].type = RawArgumentType::kI64;
         },
         "wrong batchnorm slot7 type was accepted");
  reject([](AscendStageArtifact& stage) {
           stage.arguments[6].scalar = stage.n_elements + 1;
         },
         "wrong batchnorm n_elements scalar was accepted");
  reject([](AscendStageArtifact& stage) {
           stage.arguments[5].uid = stage.arguments[0].uid;
         },
         "duplicate batchnorm pointer UID was accepted");
  reject([](AscendStageArtifact& stage) { stage.batchnorm_rank = 1; },
         "rank1 batchnorm stage was accepted");
}

void check_seed_repeatability_and_parameter_roles() {
  const std::vector<std::int64_t> dimensions = {2, 5, 2};
  const std::vector<std::int64_t> x_strides = {30, 5, 2};
  const std::vector<std::int64_t> y_strides = {40, 7, 3};
  AscendStageArtifact stage = make_stage(StorageDataType::kFloat32,
                                         dimensions,
                                         x_strides,
                                         y_strides,
                                         false);
  const auto make_buffers = [&]() {
    return std::array<std::vector<std::uint8_t>, 5>{
        std::vector<std::uint8_t>(
            storage_elements(dimensions, x_strides) * sizeof(float), 0xA5U),
        std::vector<std::uint8_t>(5U * sizeof(float), 0xA5U),
        std::vector<std::uint8_t>(5U * sizeof(float), 0xA5U),
        std::vector<std::uint8_t>(5U * sizeof(float), 0xA5U),
        std::vector<std::uint8_t>(5U * sizeof(float), 0xA5U)};
  };
  auto first = make_buffers();
  auto second = make_buffers();
  flagdnn::ascend::seed_batchnorm_inference_host_inputs(
      stage, first[0], first[1], first[2], first[3], first[4]);
  flagdnn::ascend::seed_batchnorm_inference_host_inputs(
      stage, second[0], second[1], second[2], second[3], second[4]);
  if (first != second) {
    throw std::runtime_error("batchnorm seed is not byte-identical");
  }
  std::vector<std::uint8_t> actual(
      storage_elements(dimensions, y_strides) * sizeof(float), 0xA5U);
  flagdnn::ascend::compute_batchnorm_inference_host_oracle(
      stage, first[0], first[1], first[2], first[3], first[4], actual);
  const auto reject_swap = [&](std::size_t left,
                               std::size_t right,
                               const char* role_names) {
    std::array<std::span<const std::uint8_t>, 4> parameters = {
        first[1], first[2], first[3], first[4]};
    std::swap(parameters[left], parameters[right]);
    expect_invalid(
        [&]() {
          flagdnn::ascend::validate_batchnorm_inference_host_output(
              stage,
              first[0],
              parameters[0],
              parameters[1],
              parameters[2],
              parameters[3],
              actual);
        },
        role_names);
  };
  reject_swap(0, 1, "swapped mean/inv_variance roles were accepted");
  reject_swap(0, 2, "swapped mean/scale roles were accepted");
  reject_swap(0, 3, "swapped mean/bias roles were accepted");
  reject_swap(1, 3, "swapped inv_variance/bias roles were accepted");
  reject_swap(2, 3, "swapped scale/bias roles were accepted");
}

void check_seeded_cancellation(StorageDataType type) {
  const std::vector<std::int64_t> dimensions = {2, 3, 2};
  const std::vector<std::int64_t> x_strides = {6, 2, 1};
  const std::vector<std::int64_t> y_strides = {8, 2, 1};
  AscendStageArtifact stage =
      make_stage(type, dimensions, x_strides, y_strides, false);
  std::vector<std::uint8_t> x(
      storage_elements(dimensions, x_strides) * element_size(type), 0xA5U);
  std::vector<std::uint8_t> mean(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> inv_variance(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> scale(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> bias(3U * sizeof(float), 0xA5U);
  std::vector<std::uint8_t> y(
      storage_elements(dimensions, y_strides) * element_size(type), 0xA5U);
  flagdnn::ascend::seed_batchnorm_inference_host_inputs(
      stage, x, mean, inv_variance, scale, bias);

  const float delta_unit =
      type == StorageDataType::kFloat32
          ? 1.0F / 256.0F
          : type == StorageDataType::kFloat16 ? 1.0F / 32.0F : 1.0F / 4.0F;
  bool saw_large_mean_cancellation = false;
  for (std::size_t logical = 0; logical < 12U; ++logical) {
    const std::size_t channel = (logical / 2U) % 3U;
    const float x_value =
        load_value(x, tensor_offset(logical, dimensions, x_strides), type);
    const float mean_value =
        load_value(mean, channel, StorageDataType::kFloat32);
    const float delta = x_value - mean_value;
    if (std::abs(mean_value) >= 16.0F && delta != 0.0F &&
        std::abs(delta) <= 2.0F * delta_unit) {
      saw_large_mean_cancellation = true;
    }
  }
  if (!saw_large_mean_cancellation) {
    throw std::runtime_error(
        "batchnorm seed misses dtype-representable large-mean cancellation");
  }

  flagdnn::ascend::compute_batchnorm_inference_host_oracle(
      stage, x, mean, inv_variance, scale, bias, y);
  const float tolerance = type == StorageDataType::kFloat32
                              ? 1.0e-5F
                              : type == StorageDataType::kFloat16 ? 0.125F : 1.0F;
  for (std::size_t logical = 0; logical < 12U; ++logical) {
    const std::size_t channel = (logical / 2U) % 3U;
    const float expected =
        (load_value(x, tensor_offset(logical, dimensions, x_strides), type) -
         load_value(mean, channel, StorageDataType::kFloat32)) *
            load_value(inv_variance, channel, StorageDataType::kFloat32) *
            load_value(scale, channel, StorageDataType::kFloat32) +
        load_value(bias, channel, StorageDataType::kFloat32);
    const float actual =
        load_value(y, tensor_offset(logical, dimensions, y_strides), type);
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
      throw std::runtime_error("batchnorm seeded formula mismatch");
    }
  }
}

void check_large_mean_cancellation() {
  AscendStageArtifact stage = make_stage(StorageDataType::kFloat32,
                                         {1, 3},
                                         {3, 1},
                                         {3, 1},
                                         false);
  std::vector<std::uint8_t> x(3U * sizeof(float), 0U);
  std::vector<std::uint8_t> mean(3U * sizeof(float), 0U);
  std::vector<std::uint8_t> inv(3U * sizeof(float), 0U);
  std::vector<std::uint8_t> scale(3U * sizeof(float), 0U);
  std::vector<std::uint8_t> bias(3U * sizeof(float), 0U);
  std::vector<std::uint8_t> y(3U * sizeof(float), 0xA5U);
  constexpr float kMean = 100000000.0F;
  const float adjacent =
      std::nextafter(kMean, std::numeric_limits<float>::infinity());
  for (std::size_t channel = 0; channel < 3U; ++channel) {
    store_value(x, channel, StorageDataType::kFloat32, adjacent);
    store_value(mean, channel, StorageDataType::kFloat32, kMean);
    store_value(inv, channel, StorageDataType::kFloat32, 0.125F);
    store_value(scale, channel, StorageDataType::kFloat32, 1.0F);
    store_value(bias, channel, StorageDataType::kFloat32, 0.0F);
  }
  flagdnn::ascend::compute_batchnorm_inference_host_oracle(
      stage, x, mean, inv, scale, bias, y);
  for (std::size_t channel = 0; channel < 3U; ++channel) {
    if (load_value(y, channel, StorageDataType::kFloat32) != 1.0F) {
      throw std::runtime_error("batchnorm large-mean cancellation mismatch");
    }
  }
}

struct FakePrewarmResources {
  int restore_count = 0;
  int reset_count = 0;
  int read_count = 0;
  int validate_inputs_count = 0;
  int validate_count = 0;
  std::uint8_t input = 0x11U;
  std::vector<std::uint8_t> output = {0x2AU};

  void restore_stage_inputs(int) { ++restore_count; }
  void reset_stage_output(int) {
    ++reset_count;
    output[0] = 0xA5U;
  }
  std::vector<std::uint8_t> read_stage_output(int) {
    ++read_count;
    return output;
  }
  void validate_stage_inputs_unchanged(int) {
    ++validate_inputs_count;
    if (input != 0x11U) {
      throw std::invalid_argument("prewarm input was modified");
    }
  }
  void validate_stage_output(int,
                             const std::vector<std::uint8_t>& actual) {
    ++validate_count;
    if (actual != std::vector<std::uint8_t>{0x2AU}) {
      throw std::invalid_argument("prewarm output was not written");
    }
  }
};

void check_build_time_prewarm_resets_output_before_launch() {
  FakePrewarmResources no_write;
  bool rejected = false;
  try {
    (void)flagdnn::ascend::detail::run_checked_build_time_prewarm(
        no_write, 0, []() {});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || no_write.restore_count != 1 || no_write.reset_count != 1 ||
      no_write.read_count != 1 || no_write.validate_count != 1) {
    throw std::runtime_error(
        "build-time prewarm accepted a launch that did not write output");
  }

  FakePrewarmResources writes_output;
  const std::vector<std::uint8_t> output =
      flagdnn::ascend::detail::run_checked_build_time_prewarm(
          writes_output, 0, [&writes_output]() {
            writes_output.output[0] = 0x2AU;
          });
  if (output != std::vector<std::uint8_t>{0x2AU} ||
      writes_output.restore_count != 1 || writes_output.reset_count != 1 ||
      writes_output.read_count != 1 || writes_output.validate_count != 1) {
    throw std::runtime_error(
        "build-time prewarm did not preserve its checked launch protocol");
  }
}

void check_build_time_prewarm_rejects_input_mutation() {
  FakePrewarmResources resources;
  bool rejected = false;
  try {
    (void)flagdnn::ascend::detail::run_checked_build_time_prewarm(
        resources, 0, [&resources]() {
          resources.output[0] = 0x2AU;
          resources.input = 0x12U;
        });
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected || resources.validate_inputs_count != 1) {
    throw std::runtime_error(
        "build-time prewarm accepted a launch that modified an input");
  }
}

}  // namespace

int main() {
  try {
    const auto run = [](const char* name, auto&& function) {
      try {
        function();
      } catch (const std::exception& error) {
        throw std::runtime_error(std::string(name) + ": " + error.what());
      }
    };
    run("fp32_rank2", []() {
      check_case(StorageDataType::kFloat32, {2, 3}, {3, 1}, {4, 1}, false);
    });
    run("fp16_gapped", []() {
      check_case(StorageDataType::kFloat16,
                 {2, 3, 2},
                 {20, 5, 2},
                 {25, 7, 3},
                 false);
    });
    run("bf16_channels_last", []() {
      check_case(StorageDataType::kBFloat16,
                 {2, 3, 2, 2},
                 {12, 1, 6, 3},
                 {16, 1, 8, 4},
                 true);
    });
    run("runtime_seed", check_runtime_contract_and_seeding);
    run("negative_stage", check_negative_stage_contract);
    run("nonfinite", check_nonfinite_inputs_and_actual);
    run("fp16_nonfinite", []() {
      check_low_precision_nonfinite(StorageDataType::kFloat16);
    });
    run("bf16_nonfinite", []() {
      check_low_precision_nonfinite(StorageDataType::kBFloat16);
    });
    run("fp16_rounding", []() {
      check_fractional_rounding(StorageDataType::kFloat16);
    });
    run("bf16_rounding", []() {
      check_fractional_rounding(StorageDataType::kBFloat16);
    });
    run("alias_overlap", check_alias_and_overlap_rejection);
    run("short_storage", check_short_storage_rejection);
    run("rank8_metadata", check_rank8_and_malformed_metadata);
    run("commit_source", check_commit_and_source_metadata);
    run("seed_parameter_roles", check_seed_repeatability_and_parameter_roles);
    run("seeded_cancellation_fp32", []() {
      check_seeded_cancellation(StorageDataType::kFloat32);
    });
    run("seeded_cancellation_fp16", []() {
      check_seeded_cancellation(StorageDataType::kFloat16);
    });
    run("seeded_cancellation_bf16", []() {
      check_seeded_cancellation(StorageDataType::kBFloat16);
    });
    run("large_mean_cancellation", check_large_mean_cancellation);
    run("prewarm_output_reset",
        check_build_time_prewarm_resets_output_before_launch);
    run("prewarm_input_immutability",
        check_build_time_prewarm_rejects_input_mutation);
    run("candidate_compare_fp32", []() {
      check_candidate_output_comparison(StorageDataType::kFloat32);
    });
    run("candidate_compare_fp16", []() {
      check_candidate_output_comparison(StorageDataType::kFloat16);
    });
    run("candidate_compare_bf16", []() {
      check_candidate_output_comparison(StorageDataType::kBFloat16);
    });
    run("rank5_c5_gapped", []() {
      check_case(StorageDataType::kFloat32,
                 {2, 5, 2, 1, 2},
                 {100, 16, 8, 8, 2},
                 {120, 20, 10, 10, 3},
                 false);
    });
    run("rank4_c5_channels_last", []() {
      check_case(StorageDataType::kFloat16,
                 {2, 5, 2, 2},
                 {20, 1, 10, 5},
                 {28, 1, 14, 7},
                 false);
    });
    run("rank4_c17_channels_last", []() {
      check_case(StorageDataType::kBFloat16,
                 {1, 17, 2, 2},
                 {68, 1, 34, 17},
                 {80, 1, 40, 20},
                 false);
    });
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
