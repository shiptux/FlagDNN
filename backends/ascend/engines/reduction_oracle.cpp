/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/reduction_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace flagdnn::ascend {
namespace {

[[noreturn]] void invalid_oracle(const char* message) {
  throw std::invalid_argument(message);
}

std::uint64_t tensor_offset(
    std::uint64_t logical,
    const std::array<std::int64_t, 8>& dimensions,
    const std::array<std::int64_t, 8>& strides) {
  std::uint64_t result = 0;
  for (std::size_t reversed = dimensions.size(); reversed != 0; --reversed) {
    const std::size_t axis = reversed - 1;
    if (dimensions[axis] <= 0 || strides[axis] < 0) {
      invalid_oracle("reduction oracle metadata is invalid");
    }
    const auto dimension = static_cast<std::uint64_t>(dimensions[axis]);
    const std::uint64_t coordinate = logical % dimension;
    logical /= dimension;
    const auto stride = static_cast<std::uint64_t>(strides[axis]);
    if (coordinate != 0 &&
        stride > std::numeric_limits<std::uint64_t>::max() / coordinate) {
      invalid_oracle("reduction oracle tensor offset overflows");
    }
    const std::uint64_t term = coordinate * stride;
    if (result > std::numeric_limits<std::uint64_t>::max() - term) {
      invalid_oracle("reduction oracle tensor offset overflows");
    }
    result += term;
  }
  if (logical != 0) {
    invalid_oracle("reduction oracle logical index exceeds tensor shape");
  }
  return result;
}

std::size_t storage_element_size(StorageDataType type) {
  switch (type) {
    case StorageDataType::kFloat32:
      return sizeof(float);
    case StorageDataType::kFloat16:
    case StorageDataType::kBFloat16:
      return sizeof(std::uint16_t);
    default:
      invalid_oracle("reduction oracle storage data type is invalid");
  }
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
  std::uint32_t result = sign |
                         (static_cast<std::uint32_t>(half_exponent) << 10U) |
                         (rounded >> 13U);
  if ((rounded & 0x00800000U) != 0U) {
    result = sign | (static_cast<std::uint32_t>(half_exponent + 1) << 10U);
  }
  return static_cast<std::uint16_t>(result);
}

float load_value(std::span<const std::uint8_t> input,
                 StorageDataType type,
                 std::uint64_t element) {
  const std::size_t element_size = storage_element_size(type);
  if (element >= input.size() / element_size) {
    invalid_oracle("reduction oracle input offset is out of range");
  }
  float result = 0.0F;
  if (type == StorageDataType::kFloat32) {
    std::memcpy(&result, input.data() + element * element_size, sizeof(result));
  } else {
    std::uint16_t value = 0;
    std::memcpy(&value, input.data() + element * element_size, sizeof(value));
    if (type == StorageDataType::kFloat16) {
      result = half_to_float(value);
    } else {
      const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
      std::memcpy(&result, &bits, sizeof(result));
    }
  }
  if (!std::isfinite(result)) {
    invalid_oracle("reduction oracle input must be finite");
  }
  return result;
}

void store_value(std::span<std::uint8_t> output,
                 StorageDataType type,
                 std::uint64_t element,
                 float value) {
  if (!std::isfinite(value)) {
    invalid_oracle("reduction oracle result must be finite");
  }
  const std::size_t element_size = storage_element_size(type);
  if (element >= output.size() / element_size) {
    invalid_oracle("reduction oracle output offset is out of range");
  }
  if (type == StorageDataType::kFloat32) {
    std::memcpy(output.data() + element * element_size, &value, sizeof(value));
  } else {
    std::uint16_t encoded = 0;
    if (type == StorageDataType::kFloat16) {
      encoded = float_to_half(value);
      if ((encoded & 0x7c00U) == 0x7c00U) {
        invalid_oracle("reduction oracle result must be finite");
      }
    } else {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      bits += 0x00007fffU + ((bits >> 16U) & 1U);
      encoded = static_cast<std::uint16_t>(bits >> 16U);
      if ((encoded & 0x7f80U) == 0x7f80U) {
        invalid_oracle("reduction oracle result must be finite");
      }
    }
    std::memcpy(output.data() + element * element_size,
                &encoded,
                sizeof(encoded));
  }
}

}  // namespace

void compute_reduction_host_oracle(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> input,
    std::span<std::uint8_t> output) {
  if (stage.kernel_family != KernelFamily::kReduction ||
      stage.input_count != 1 || stage.n_elements <= 0 ||
      stage.reduction_outer <= 0 || stage.reduction_size <= 0 ||
      stage.reduction_inner <= 0 ||
      stage.tensor_storage_data_types.size() != 2 ||
      stage.input_type(0) != stage.output_type() ||
      (stage.output_type() != StorageDataType::kFloat32 &&
       stage.output_type() != StorageDataType::kFloat16 &&
       stage.output_type() != StorageDataType::kBFloat16)) {
    invalid_oracle("reduction host oracle ABI is invalid");
  }
  if (stage.operation != "reduction_sum" &&
      stage.operation != "reduction_avg" &&
      stage.operation != "reduction_mul") {
    invalid_oracle("reduction host oracle operation is invalid");
  }
  for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
    const std::uint64_t output_logical = static_cast<std::uint64_t>(logical);
    const std::uint64_t outer =
        output_logical / static_cast<std::uint64_t>(stage.reduction_inner);
    const std::uint64_t inner =
        output_logical % static_cast<std::uint64_t>(stage.reduction_inner);
    double accumulator = stage.operation == "reduction_mul" ? 1.0 : 0.0;
    for (std::int64_t reduction = 0; reduction < stage.reduction_size;
         ++reduction) {
      const std::uint64_t input_logical =
          outer * static_cast<std::uint64_t>(stage.reduction_size) *
              static_cast<std::uint64_t>(stage.reduction_inner) +
          static_cast<std::uint64_t>(reduction) *
              static_cast<std::uint64_t>(stage.reduction_inner) +
          inner;
      const float value = load_value(
          input,
          stage.input_type(0),
          tensor_offset(input_logical,
                        stage.reduction_input_dimensions,
                        stage.reduction_input_strides));
      if (stage.operation == "reduction_mul") {
        accumulator *= static_cast<double>(value);
      } else {
        accumulator += static_cast<double>(value);
      }
    }
    if (stage.operation == "reduction_avg") {
      accumulator /= static_cast<double>(stage.reduction_size);
    }
    if (!std::isfinite(accumulator) ||
        accumulator > static_cast<double>(std::numeric_limits<float>::max()) ||
        accumulator < -static_cast<double>(std::numeric_limits<float>::max())) {
      invalid_oracle("reduction oracle result must be finite");
    }
    store_value(output,
                stage.output_type(),
                tensor_offset(output_logical,
                              stage.reduction_output_dimensions,
                              stage.output_strides),
                static_cast<float>(accumulator));
  }
}

std::size_t reduction_kernel_input_count() noexcept { return 1U; }

std::size_t reduction_output_argument_index() noexcept { return 1U; }

std::size_t reduction_runtime_argument_count() noexcept { return 3U; }

const char* reduction_output_argument_name() noexcept { return "output_ptr"; }

void validate_reduction_stage_runtime_contract(
    const AscendStageArtifact& stage) {
  if (stage.kernel_family != KernelFamily::kReduction ||
      stage.input_count != reduction_kernel_input_count() ||
      stage.arguments.size() != reduction_runtime_argument_count() ||
      stage.arguments[0].index != 0U ||
      stage.arguments[0].name != "input_ptr" ||
      stage.arguments[0].type != RawArgumentType::kPointer ||
      stage.arguments[1].index != reduction_output_argument_index() ||
      stage.arguments[1].name != reduction_output_argument_name() ||
      stage.arguments[1].type != RawArgumentType::kPointer ||
      stage.arguments[2].index != 2U ||
      stage.arguments[2].name != "n_elements" ||
      stage.arguments[2].source != ArgumentSourceKind::kScalar ||
      stage.arguments[2].type != RawArgumentType::kI32) {
    invalid_oracle("reduction stage runtime ABI is invalid");
  }
  const auto* n_elements = std::get_if<std::int32_t>(&stage.arguments[2].scalar);
  if (n_elements == nullptr || *n_elements != stage.n_elements) {
    invalid_oracle("reduction stage n_elements scalar is invalid");
  }
}

void seed_reduction_host_input(
    const AscendStageArtifact& stage,
    std::span<std::uint8_t> input) {
  validate_reduction_stage_runtime_contract(stage);
  if (stage.reduction_outer <= 0 || stage.reduction_size <= 0 ||
      stage.reduction_inner <= 0) {
    invalid_oracle("reduction seed metadata is invalid");
  }
  const auto outer = static_cast<std::uint64_t>(stage.reduction_outer);
  const auto reduction_size =
      static_cast<std::uint64_t>(stage.reduction_size);
  const auto inner = static_cast<std::uint64_t>(stage.reduction_inner);
  if (outer > std::numeric_limits<std::uint64_t>::max() / reduction_size ||
      outer * reduction_size >
          std::numeric_limits<std::uint64_t>::max() / inner) {
    invalid_oracle("reduction seed logical size overflows");
  }
  const std::uint64_t logical_count = outer * reduction_size * inner;
  for (std::uint64_t logical = 0; logical < logical_count; ++logical) {
    const std::uint64_t reduction = (logical / inner) % reduction_size;
    const std::uint64_t outer_coordinate = logical / (reduction_size * inner);
    const std::uint64_t inner_coordinate = logical % inner;
    const std::uint64_t output_code =
        (outer_coordinate * inner + inner_coordinate) % 8U;
    const float additive_code =
        0.25F + static_cast<float>(output_code) * 0.125F;
    constexpr float kMultiplicativeCodes[] = {
        0.5F, -0.5F, 1.0F, -1.0F, 2.0F, -2.0F, 0.5F, -0.5F};
    float value = 0.0F;
    if (stage.operation == "reduction_mul") {
      value = reduction == 0U
                  ? kMultiplicativeCodes[output_code]
                  : (reduction % 2U == 0U ? 1.0F : -1.0F);
    } else if (stage.operation == "reduction_avg") {
      value = additive_code +
              (reduction % 2U == 0U ? 0.125F : -0.125F);
    } else {
      value = reduction == 0U
                  ? additive_code
                  : (reduction % 2U == 0U ? -0.5F : 0.5F);
    }
    store_value(input,
                stage.input_type(0),
                tensor_offset(logical,
                              stage.reduction_input_dimensions,
                              stage.reduction_input_strides),
                value);
  }
}

void validate_reduction_host_output(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> input,
    std::span<const std::uint8_t> actual) {
  validate_reduction_stage_runtime_contract(stage);
  const std::size_t element_size = storage_element_size(stage.output_type());
  if (actual.size() % element_size != 0U) {
    invalid_oracle("reduction output has a partial storage element");
  }
  std::vector<std::uint8_t> expected(actual.size(), 0xA5U);
  compute_reduction_host_oracle(stage, input, expected);
  std::vector<bool> written(actual.size() / element_size, false);
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
        stage.reduction_output_dimensions,
        stage.output_strides);
    const float expected_value =
        load_value(expected, stage.output_type(), offset);
    const float actual_value = load_value(actual, stage.output_type(), offset);
    const float tolerance =
        absolute_base + relative_tolerance * std::abs(expected_value);
    if (std::abs(actual_value - expected_value) > tolerance) {
      invalid_oracle("reduction candidate differs from the host oracle");
    }
    if (written.at(static_cast<std::size_t>(offset))) {
      invalid_oracle("reduction output metadata aliases logical elements");
    }
    written[static_cast<std::size_t>(offset)] = true;
  }
  for (std::size_t element = 0; element < written.size(); ++element) {
    if (written[element]) {
      continue;
    }
    for (std::size_t byte = 0; byte < element_size; ++byte) {
      if (actual[element * element_size + byte] != 0xA5U) {
        invalid_oracle("reduction candidate modified output padding");
      }
    }
  }
}

void commit_reduction_host_output(
    const AscendStageArtifact& stage,
    std::span<const std::uint8_t> actual,
    std::span<std::uint8_t> shadow) {
  validate_reduction_stage_runtime_contract(stage);
  if (actual.size() != shadow.size()) {
    invalid_oracle("reduction output commit size differs");
  }
  std::copy(actual.begin(), actual.end(), shadow.begin());
}

}  // namespace flagdnn::ascend
