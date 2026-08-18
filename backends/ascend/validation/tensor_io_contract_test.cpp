/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/tensor_io.hpp"

#include <flagdnn/flagdnn.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;

struct Tensor {
  flagdnnDataType_t data_type = FLAGDNN_DATA_BOOLEAN;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  std::size_t binding_byte_offset = 0;
};

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template <typename Callback>
void require_failure(Callback&& callback, std::string_view needle) {
  try {
    callback();
  } catch (const std::exception& error) {
    require(std::string_view(error.what()).find(needle) !=
                std::string_view::npos,
            "tensor I/O failure did not identify the violated contract");
    return;
  }
  throw std::runtime_error("tensor I/O contract violation was accepted");
}

}  // namespace

int main() {
  try {
    const Tensor scalar{
        FLAGDNN_DATA_FLOAT32,
        {},
        {},
        12,
    };
    require(tensor_io::element_count(scalar) == 1,
            "rank-0 tensor must contain one logical element");
    require(tensor_io::storage_element_count(scalar) == 1,
            "rank-0 tensor must occupy one storage element");
    require(tensor_io::encoded_byte_count(scalar) == sizeof(float),
            "rank-0 FP32 tensor byte extent is wrong");
    require(tensor_io::allocation_byte_count(scalar) ==
                12U + sizeof(float),
            "rank-0 tensor binding offset is missing from allocation extent");
    require(tensor_io::physical_offset(0, scalar) == 0,
            "rank-0 tensor physical offset must be zero");
    const std::vector<float> scalar_logical = {3.25F};
    const std::vector<float> scalar_physical =
        tensor_io::scatter(scalar_logical, scalar);
    require(scalar_physical == scalar_logical,
            "rank-0 tensor scatter changed the scalar");
    require(tensor_io::gather(scalar_physical, scalar) == scalar_logical,
            "rank-0 tensor gather changed the scalar");
    tensor_io::require_padding_unchanged(
        "contract", scalar_physical, scalar);
    require_failure(
        [&] { (void)tensor_io::physical_offset(1, scalar); },
        "out of range");

    const Tensor tensor{
        FLAGDNN_DATA_BOOLEAN,
        {2, 2},
        {5, 2},
        7,
    };
    require(tensor_io::data_type_size(FLAGDNN_DATA_BOOLEAN) == 1,
            "BOOLEAN must use exactly one byte");
    require(tensor_io::element_count(tensor) == 4,
            "logical BOOLEAN extent is wrong");
    require(tensor_io::storage_element_count(tensor) == 8,
            "gapped BOOLEAN storage extent is wrong");
    require(tensor_io::encoded_byte_count(tensor) == 8,
            "BOOLEAN encoded extent is wrong");
    require(tensor_io::allocation_byte_count(tensor) == 15,
            "BOOLEAN binding offset is missing from allocation extent");

    const std::vector<float> logical = {0.0F, 1.0F, 1.0F, 0.0F};
    const std::vector<float> physical = tensor_io::scatter(logical, tensor);
    const std::vector<std::uint8_t> encoded =
        tensor_io::encode(physical, tensor.data_type);
    require(encoded.size() == 8, "BOOLEAN encoding is not byte-addressed");
    const std::vector<std::size_t> occupied = {0, 2, 5, 7};
    for (std::size_t index = 0; index < encoded.size(); ++index) {
      const bool is_occupied = index == occupied[0] || index == occupied[1] ||
                               index == occupied[2] || index == occupied[3];
      if (is_occupied) {
        require(encoded[index] <= 1,
                "BOOLEAN logical data is not canonical 0/1");
      } else {
        require(encoded[index] == tensor_io::kBooleanPaddingSentinel,
                "BOOLEAN padding was not initialized to 0xA5");
      }
    }

    const std::vector<float> decoded = tensor_io::decode_storage(
        "contract", encoded, tensor);
    require(tensor_io::gather(decoded, tensor) == logical,
            "gapped BOOLEAN round trip changed truth values");
    tensor_io::require_padding_unchanged("contract", decoded, tensor);

    std::vector<std::uint8_t> corrupt = encoded;
    corrupt[1] = 0;
    corrupt[0] = 7;
    require_failure(
        [&] { (void)tensor_io::decode_storage("contract", corrupt, tensor); },
        "padding");

    corrupt = encoded;
    corrupt[0] = 7;
    require_failure(
        [&] { (void)tensor_io::decode_storage("contract", corrupt, tensor); },
        "canonical");

    corrupt = encoded;
    corrupt[7] = tensor_io::kBooleanPaddingSentinel;
    require_failure(
        [&] { (void)tensor_io::decode_storage("contract", corrupt, tensor); },
        "canonical");

    require_failure(
        [&] {
          const std::vector<float> noncanonical = {0.0F, 2.0F};
          (void)tensor_io::encode(noncanonical, FLAGDNN_DATA_BOOLEAN);
        },
        "canonical");

    std::cout << "Ascend BOOLEAN tensor I/O contract: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Ascend BOOLEAN tensor I/O contract failed: "
              << error.what() << '\n';
    return 1;
  }
}
