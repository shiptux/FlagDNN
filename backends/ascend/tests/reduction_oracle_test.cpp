/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/reduction_oracle.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using flagdnn::ascend::AscendStageArtifact;
using flagdnn::ascend::ArgumentSource;
using flagdnn::ascend::ArgumentSourceKind;
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

void store_value(std::vector<std::uint8_t>& bytes,
                 std::size_t element,
                 StorageDataType type,
                 float value) {
  const std::size_t size = element_size(type);
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

float load_value(const std::vector<std::uint8_t>& bytes,
                 std::size_t element,
                 StorageDataType type) {
  const std::size_t size = element_size(type);
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
  const std::uint32_t sign = static_cast<std::uint32_t>(encoded & 0x8000U) << 16U;
  const std::uint32_t exponent = (encoded >> 10U) & 0x1FU;
  const std::uint32_t mantissa = encoded & 0x03FFU;
  if (exponent == 0) {
    return std::bit_cast<float>(sign);
  }
  return std::bit_cast<float>(
      sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
}

float load_float(const std::vector<std::uint8_t>& bytes, std::size_t element) {
  float result = 0.0F;
  std::memcpy(&result, bytes.data() + element * sizeof(float), sizeof(result));
  return result;
}

AscendStageArtifact make_stage(const std::string& operation,
                               StorageDataType type = StorageDataType::kFloat32) {
  AscendStageArtifact stage;
  stage.operation = operation;
  stage.kernel_family = KernelFamily::kReduction;
  stage.tensor_storage_data_types = {type, type};
  stage.input_count = 1;
  stage.n_elements = 6;
  stage.reduction_rank = 3;
  stage.reduction_output_rank = 3;
  stage.reduction_axis = 1;
  stage.reduction_keep_dimensions = 1;
  stage.reduction_outer = 2;
  stage.reduction_size = 4;
  stage.reduction_inner = 3;
  stage.reduction_input_dimensions = {1, 1, 1, 1, 1, 2, 4, 3};
  stage.reduction_input_strides = {0, 0, 0, 0, 0, 20, 4, 1};
  stage.reduction_output_dimensions = {1, 1, 1, 1, 1, 2, 1, 3};
  stage.output_strides = {0, 0, 0, 0, 0, 5, 5, 1};
  ArgumentSource input_argument;
  input_argument.index = 0;
  input_argument.name = "input_ptr";
  input_argument.source = ArgumentSourceKind::kBinding;
  input_argument.type = RawArgumentType::kPointer;
  ArgumentSource output_argument = input_argument;
  output_argument.index = 1;
  output_argument.name = "output_ptr";
  ArgumentSource n_elements_argument;
  n_elements_argument.index = 2;
  n_elements_argument.name = "n_elements";
  n_elements_argument.source = ArgumentSourceKind::kScalar;
  n_elements_argument.type = RawArgumentType::kI32;
  n_elements_argument.scalar = stage.n_elements;
  stage.arguments = {input_argument, output_argument, n_elements_argument};
  return stage;
}

float input_value(std::size_t outer,
                  std::size_t reduction,
                  std::size_t inner) {
  return 0.5F + static_cast<float>((outer + reduction + inner) % 4U);
}

void check_modes_and_padding() {
  for (const StorageDataType type :
       {StorageDataType::kFloat32, StorageDataType::kFloat16,
        StorageDataType::kBFloat16}) {
   for (const std::string operation :
        {"reduction_sum", "reduction_avg", "reduction_mul"}) {
    AscendStageArtifact stage = make_stage(operation, type);
    const std::size_t size = element_size(type);
    std::vector<std::uint8_t> input(35U * size, 0U);
    for (std::size_t outer = 0; outer < 2; ++outer) {
      for (std::size_t reduction = 0; reduction < 4; ++reduction) {
        for (std::size_t inner = 0; inner < 3; ++inner) {
          store_value(input,
                      outer * 20U + reduction * 4U + inner,
                      type,
                      input_value(outer, reduction, inner));
        }
      }
    }
    std::vector<std::uint8_t> output(8U * size, 0xA5U);
    flagdnn::ascend::compute_reduction_host_oracle(stage, input, output);
    std::array<bool, 8> written{};
    for (std::size_t outer = 0; outer < 2; ++outer) {
      for (std::size_t inner = 0; inner < 3; ++inner) {
        float expected = operation == "reduction_mul" ? 1.0F : 0.0F;
        for (std::size_t reduction = 0; reduction < 4; ++reduction) {
          const float value = input_value(outer, reduction, inner);
          if (operation == "reduction_mul") {
            expected *= value;
          } else {
            expected += value;
          }
        }
        if (operation == "reduction_avg") {
          expected /= 4.0F;
        }
        const std::size_t output_offset = outer * 5U + inner;
        written[output_offset] = true;
        const float observed = load_value(output, output_offset, type);
        const float tolerance = type == StorageDataType::kFloat32 ? 1.0e-6F : 5.0e-2F;
        if (std::abs(observed - expected) > tolerance * std::max(1.0F, std::abs(expected))) {
          throw std::runtime_error("reduction host oracle value mismatch");
        }
      }
    }
    for (std::size_t element = 0; element < written.size(); ++element) {
      if (!written[element] &&
          !std::all_of(output.begin() + element * size,
                       output.begin() + (element + 1U) * size,
                       [](std::uint8_t byte) { return byte == 0xA5U; })) {
        throw std::runtime_error("reduction host oracle overwrote padding");
      }
    }

    const std::vector<std::uint8_t> committed_shadow = output;
    std::fill(output.begin(), output.end(), 0U);
    std::copy(committed_shadow.begin(), committed_shadow.end(), output.begin());
    if (output != committed_shadow) {
      throw std::runtime_error("mixed DAG output shadow was not restored");
    }
   }
  }
}

void check_rank0() {
  AscendStageArtifact stage = make_stage("reduction_sum");
  stage.n_elements = 1;
  stage.reduction_rank = 1;
  stage.reduction_output_rank = 0;
  stage.reduction_axis = 0;
  stage.reduction_keep_dimensions = 0;
  stage.reduction_outer = 1;
  stage.reduction_size = 4;
  stage.reduction_inner = 1;
  stage.reduction_input_dimensions = {1, 1, 1, 1, 1, 1, 1, 4};
  stage.reduction_input_strides = {0, 0, 0, 0, 0, 0, 0, 1};
  stage.reduction_output_dimensions = {1, 1, 1, 1, 1, 1, 1, 1};
  stage.output_strides = {};
  std::vector<std::uint8_t> input(4U * sizeof(float));
  for (std::size_t index = 0; index < 4; ++index) {
    store_value(input, index, StorageDataType::kFloat32,
                static_cast<float>(index + 1U));
  }
  std::vector<std::uint8_t> output(sizeof(float), 0xA5U);
  flagdnn::ascend::compute_reduction_host_oracle(stage, input, output);
  if (load_float(output, 0) != 10.0F) {
    throw std::runtime_error("rank0 reduction host oracle mismatch");
  }
}

void check_nonfinite_rejection() {
  AscendStageArtifact stage = make_stage("reduction_sum");
  std::vector<std::uint8_t> input(35U * sizeof(float), 0U);
  for (std::size_t element = 0; element < 35; ++element) {
    store_value(input, element, StorageDataType::kFloat32, 1.0F);
  }
  std::vector<std::uint8_t> output(8U * sizeof(float), 0xA5U);
  store_value(input, 0, StorageDataType::kFloat32,
              std::numeric_limits<float>::infinity());
  try {
    flagdnn::ascend::compute_reduction_host_oracle(stage, input, output);
    throw std::runtime_error("nonfinite reduction input was accepted");
  } catch (const std::invalid_argument&) {
  }

  stage.operation = "reduction_mul";
  std::fill(input.begin(), input.end(), 0U);
  for (std::size_t element = 0; element < 35; ++element) {
    store_value(input, element, StorageDataType::kFloat32,
                std::numeric_limits<float>::max());
  }
  try {
    flagdnn::ascend::compute_reduction_host_oracle(stage, input, output);
    throw std::runtime_error("overflowing reduction product was accepted");
  } catch (const std::invalid_argument&) {
  }
}

void check_engine_reduction_contract() {
  AscendStageArtifact stage = make_stage("reduction_mul");
  ArgumentSource input_argument;
  input_argument.index = 0;
  input_argument.name = "input_ptr";
  input_argument.source = ArgumentSourceKind::kBinding;
  input_argument.type = RawArgumentType::kPointer;
  ArgumentSource output_argument = input_argument;
  output_argument.index = 1;
  output_argument.name = "output_ptr";
  ArgumentSource n_elements_argument;
  n_elements_argument.index = 2;
  n_elements_argument.name = "n_elements";
  n_elements_argument.source = ArgumentSourceKind::kScalar;
  n_elements_argument.type = RawArgumentType::kI32;
  n_elements_argument.scalar = stage.n_elements;
  stage.arguments = {input_argument, output_argument, n_elements_argument};

  if (flagdnn::ascend::reduction_kernel_input_count() != 1U ||
      flagdnn::ascend::reduction_output_argument_index() != 1U ||
      flagdnn::ascend::reduction_runtime_argument_count() != 3U ||
      std::string(flagdnn::ascend::reduction_output_argument_name()) !=
          "output_ptr") {
    throw std::runtime_error("reduction engine ABI routing mismatch");
  }
  flagdnn::ascend::validate_reduction_stage_runtime_contract(stage);
  std::vector<std::uint8_t> input(35U * sizeof(float), 0xA5U);
  flagdnn::ascend::seed_reduction_host_input(stage, input);
  for (std::size_t outer = 0; outer < 2; ++outer) {
    for (std::size_t reduction = 0; reduction < 4; ++reduction) {
      for (std::size_t inner = 0; inner < 3; ++inner) {
        if (std::abs(load_float(
                input, outer * 20U + reduction * 4U + inner)) !=
            (reduction == 0U
                 ? std::abs(load_float(input, outer * 20U + inner))
                 : 1.0F)) {
          throw std::runtime_error("reduction MUL seed is not finite-safe");
        }
      }
    }
  }

  std::vector<std::uint8_t> actual(8U * sizeof(float), 0xA5U);
  flagdnn::ascend::compute_reduction_host_oracle(stage, input, actual);
  flagdnn::ascend::validate_reduction_host_output(stage, input, actual);
  std::vector<std::uint8_t> accumulator_only = actual;
  store_value(accumulator_only, 0, StorageDataType::kFloat32, 1.0F);
  try {
    flagdnn::ascend::validate_reduction_host_output(
        stage, input, accumulator_only);
    throw std::runtime_error("MUL accumulator-only output was accepted");
  } catch (const std::invalid_argument&) {
  }
  std::vector<std::uint8_t> tolerant = actual;
  store_value(tolerant,
              0,
              StorageDataType::kFloat32,
              load_float(tolerant, 0) + 5.0e-6F);
  flagdnn::ascend::validate_reduction_host_output(stage, input, tolerant);
  std::vector<std::uint8_t> nonfinite = actual;
  store_value(nonfinite,
              0,
              StorageDataType::kFloat32,
              std::numeric_limits<float>::quiet_NaN());
  try {
    flagdnn::ascend::validate_reduction_host_output(stage, input, nonfinite);
    throw std::runtime_error("nonfinite reduction output was accepted");
  } catch (const std::invalid_argument&) {
  }
  std::vector<std::uint8_t> shadow(actual.size(), 0U);
  flagdnn::ascend::commit_reduction_host_output(stage, actual, shadow);
  if (shadow != actual) {
    throw std::runtime_error("reduction virtual shadow commit mismatch");
  }

  AscendStageArtifact sum_stage = make_stage("reduction_sum");
  std::vector<std::uint8_t> sum_input(35U * sizeof(float), 0xA5U);
  flagdnn::ascend::seed_reduction_host_input(sum_stage, sum_input);
  std::vector<std::uint8_t> sum_output(8U * sizeof(float), 0xA5U);
  flagdnn::ascend::compute_reduction_host_oracle(
      sum_stage, sum_input, sum_output);
  std::array<float, 6> codes{};
  for (std::size_t outer = 0; outer < 2; ++outer) {
    for (std::size_t inner = 0; inner < 3; ++inner) {
      codes[outer * 3U + inner] =
          load_float(sum_output, outer * 5U + inner);
    }
  }
  std::sort(codes.begin(), codes.end());
  if (std::adjacent_find(codes.begin(), codes.end()) != codes.end()) {
    throw std::runtime_error("reduction seed does not encode output position");
  }
  std::vector<std::uint8_t> corrupted = actual;
  store_value(corrupted,
              0,
              StorageDataType::kFloat32,
              load_float(corrupted, 0) + 1.0F);
  try {
    flagdnn::ascend::validate_reduction_host_output(stage, input, corrupted);
    throw std::runtime_error("reduction logical output tamper was accepted");
  } catch (const std::invalid_argument&) {
  }
  corrupted = actual;
  corrupted[3U * sizeof(float)] ^= 1U;
  try {
    flagdnn::ascend::validate_reduction_host_output(stage, input, corrupted);
    throw std::runtime_error("reduction padding tamper was accepted");
  } catch (const std::invalid_argument&) {
  }

  stage = make_stage("reduction_mul", StorageDataType::kBFloat16);
  stage.n_elements = 1;
  stage.reduction_rank = 1;
  stage.reduction_output_rank = 0;
  stage.reduction_axis = 0;
  stage.reduction_keep_dimensions = 0;
  stage.reduction_outer = 1;
  stage.reduction_size = 65537;
  stage.reduction_inner = 1;
  stage.reduction_input_dimensions = {1, 1, 1, 1, 1, 1, 1, 65537};
  stage.reduction_input_strides = {0, 0, 0, 0, 0, 0, 0, 1};
  stage.reduction_output_dimensions = {1, 1, 1, 1, 1, 1, 1, 1};
  stage.output_strides = {};
  stage.arguments[2].scalar = stage.n_elements;
  std::vector<std::uint8_t> long_input(65537U * sizeof(std::uint16_t), 0xA5U);
  flagdnn::ascend::seed_reduction_host_input(stage, long_input);
  std::vector<std::uint8_t> long_output(sizeof(std::uint16_t), 0xA5U);
  flagdnn::ascend::compute_reduction_host_oracle(stage, long_input, long_output);
  flagdnn::ascend::validate_reduction_host_output(
      stage, long_input, long_output);

  stage = make_stage("reduction_sum");
  stage.n_elements = 1;
  stage.reduction_rank = 1;
  stage.reduction_output_rank = 0;
  stage.reduction_axis = 0;
  stage.reduction_keep_dimensions = 0;
  stage.reduction_outer = 1;
  stage.reduction_size = 1048577;
  stage.reduction_inner = 1;
  stage.reduction_input_dimensions = {1, 1, 1, 1, 1, 1, 1, 1048577};
  stage.reduction_input_strides = {0, 0, 0, 0, 0, 0, 0, 1};
  stage.reduction_output_dimensions = {1, 1, 1, 1, 1, 1, 1, 1};
  stage.output_strides = {};
  stage.arguments[2].scalar = stage.n_elements;
  std::vector<std::uint8_t> long_sum_input(1048577U * sizeof(float), 0xA5U);
  flagdnn::ascend::seed_reduction_host_input(stage, long_sum_input);
  std::vector<std::uint8_t> wrong_sum_output(sizeof(float), 0U);
  try {
    flagdnn::ascend::validate_reduction_host_output(
        stage, long_sum_input, wrong_sum_output);
    throw std::runtime_error("long SUM zero output was accepted");
  } catch (const std::invalid_argument&) {
  }

  stage = make_stage("reduction_avg", StorageDataType::kBFloat16);
  stage.n_elements = 1;
  stage.reduction_rank = 1;
  stage.reduction_output_rank = 0;
  stage.reduction_axis = 0;
  stage.reduction_keep_dimensions = 0;
  stage.reduction_outer = 1;
  stage.reduction_size = 1025;
  stage.reduction_inner = 1;
  stage.reduction_input_dimensions = {1, 1, 1, 1, 1, 1, 1, 1025};
  stage.reduction_input_strides = {0, 0, 0, 0, 0, 0, 0, 1};
  stage.reduction_output_dimensions = {1, 1, 1, 1, 1, 1, 1, 1};
  stage.output_strides = {};
  stage.arguments[2].scalar = stage.n_elements;
  std::vector<std::uint8_t> long_avg_input(1025U * sizeof(std::uint16_t), 0xA5U);
  flagdnn::ascend::seed_reduction_host_input(stage, long_avg_input);
  std::vector<std::uint8_t> wrong_avg_output(sizeof(std::uint16_t), 0U);
  try {
    flagdnn::ascend::validate_reduction_host_output(
        stage, long_avg_input, wrong_avg_output);
    throw std::runtime_error("long AVG zero output was accepted");
  } catch (const std::invalid_argument&) {
  }

  AscendStageArtifact overlapping = make_stage("reduction_sum");
  overlapping.output_strides = {};
  std::vector<std::uint8_t> overlapping_input(35U * sizeof(float), 0xA5U);
  flagdnn::ascend::seed_reduction_host_input(overlapping, overlapping_input);
  std::vector<std::uint8_t> overlapping_output(8U * sizeof(float), 0xA5U);
  flagdnn::ascend::compute_reduction_host_oracle(
      overlapping, overlapping_input, overlapping_output);
  try {
    flagdnn::ascend::validate_reduction_host_output(
        overlapping, overlapping_input, overlapping_output);
    throw std::runtime_error("overlapping reduction output was accepted");
  } catch (const std::invalid_argument&) {
  }

  stage = make_stage("reduction_avg", StorageDataType::kFloat16);
  stage.n_elements = 2;
  stage.reduction_rank = 2;
  stage.reduction_output_rank = 1;
  stage.reduction_axis = 0;
  stage.reduction_keep_dimensions = 0;
  stage.reduction_outer = 1;
  stage.reduction_size = 16777217;
  stage.reduction_inner = 2;
  stage.reduction_input_dimensions = {1, 1, 1, 1, 1, 1, 16777217, 2};
  stage.reduction_input_strides = {0, 0, 0, 0, 0, 0, 2, 1};
  stage.reduction_output_dimensions = {1, 1, 1, 1, 1, 1, 1, 2};
  stage.output_strides = {0, 0, 0, 0, 0, 0, 0, 1};
  stage.arguments[2].scalar = stage.n_elements;
  std::vector<std::uint8_t> huge_avg_input(
      16777217U * 2U * sizeof(std::uint16_t), 0xA5U);
  flagdnn::ascend::seed_reduction_host_input(stage, huge_avg_input);
  std::vector<std::uint8_t> huge_avg_output(
      2U * sizeof(std::uint16_t), 0xA5U);
  flagdnn::ascend::compute_reduction_host_oracle(
      stage, huge_avg_input, huge_avg_output);
  if (std::abs(load_value(
          huge_avg_output, 1, StorageDataType::kFloat16) - 0.375F) > 1.0e-3F) {
    throw std::runtime_error("large AVG host oracle lost ideal accumulation");
  }
}

}  // namespace

int main() {
  check_modes_and_padding();
  check_rank0();
  check_nonfinite_rejection();
  check_engine_reduction_contract();
  return 0;
}
