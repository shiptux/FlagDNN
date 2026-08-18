/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_TENSOR_RUNNER_SUPPORT_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_TENSOR_RUNNER_SUPPORT_HPP_

#include "common/layout.hpp"
#include "common/matmul.hpp"
#include "common/reduction.hpp"
#include "functional/accuracy.hpp"
#include "functional/binding_address.hpp"
#include "hip_driver.hpp"
#include "tensor_io.hpp"
#include "tensor_reference.hpp"

#include <flagdnn/flagdnn.hpp>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing::hygon_functional::tensor {

namespace hv = validation::hygon;
namespace io = validation::hygon::tensor_io;

inline constexpr int kSkipReturnCode = 77;

class TemporaryCache {
public:
  explicit TemporaryCache(std::string_view family) {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           ("flagdnn-hygon-" + std::string(family) + "-XXXXXX"))
                              .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char *created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryCache() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

class ReferenceExecutable final : public TestExecutable {
public:
  ReferenceExecutable(hv::HipdnnTensorOperation operation,
                      std::vector<hv::ReferenceTensor> tensors)
      : plan_(std::move(operation), std::move(tensors)) {}

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return plan_.workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    plan_.execute(bindings, workspace, workspace_size, stream);
  }

private:
  hv::HipdnnTensorPlan plan_;
};

inline hv::HipdnnTensorOperation
layout_operation(const LayoutTestCase &test_case) {
  switch (test_case.operation) {
  case LayoutOperation::kReshape:
    return hv::make_hipdnn_tensor_unavailable(
        "hipdnnTransformTensor reshape is not an exact validated primitive");
  case LayoutOperation::kTranspose:
    return hv::make_hipdnn_tensor_unavailable(
        "hipdnnTransformTensor does not provide a validated exact transpose");
  case LayoutOperation::kSlice:
    return hv::make_hipdnn_slice_operation(test_case.slices,
                                           test_case.slice_strides);
  }
  return hv::make_hipdnn_tensor_unavailable(
      "layout operation has no exact hipDNN primitive");
}

inline std::string_view layout_operation_name(LayoutOperation operation) {
  switch (operation) {
  case LayoutOperation::kReshape:
    return "reshape";
  case LayoutOperation::kTranspose:
    return "transpose";
  case LayoutOperation::kSlice:
    return "slice";
  }
  return "layout_unknown";
}

inline hv::HipdnnTensorOperation matmul_operation() {
  return hv::make_hipdnn_tensor_unavailable(
      "hipDNN exposes no exact primitive MatMul reference on this stack");
}

inline TestTensor dense_test_tensor(const TestTensor &tensor) {
  const hv::ReferenceTensor dense =
      hv::dense_reference_tensor(hv::as_reference_tensor(tensor));
  return {dense.uid, dense.data_type, dense.dimensions, dense.strides,
          dense.binding_byte_offset};
}

inline std::vector<hv::ReferenceTensor>
layout_reference_tensors(const LayoutTestCase &test_case) {
  return {
      hv::as_reference_tensor(test_case.input),
      hv::dense_reference_tensor(hv::as_reference_tensor(test_case.output))};
}

inline std::vector<hv::ReferenceTensor>
reduction_reference_tensors(const ReductionTestCase &test_case) {
  return {hv::as_reference_tensor(test_case.input),
          hv::as_reference_tensor(test_case.output)};
}

inline std::vector<hv::ReferenceTensor>
matmul_reference_tensors(const MatmulTestCase &test_case) {
  return {hv::as_reference_tensor(test_case.a),
          hv::as_reference_tensor(test_case.b),
          hv::as_reference_tensor(test_case.output)};
}

inline std::size_t logical_element_count(const TestTensor &tensor) {
  return tensor.dimensions.empty() ? 1 : io::element_count(tensor);
}

inline std::size_t storage_element_count(const TestTensor &tensor) {
  return tensor.dimensions.empty() ? 1 : io::storage_element_count(tensor);
}

inline std::vector<float> gather(std::span<const float> physical,
                                 const TestTensor &tensor) {
  if (tensor.dimensions.empty()) {
    if (physical.empty()) {
      throw std::invalid_argument("scalar physical storage is empty");
    }
    return {physical.front()};
  }
  return io::gather(physical, tensor);
}

inline void require_padding_unchanged(std::string_view provider,
                                      std::span<const float> physical,
                                      const TestTensor &tensor) {
  if (!tensor.dimensions.empty()) {
    io::require_padding_unchanged(provider, physical, tensor);
  }
}

inline std::vector<float> make_input(std::size_t count,
                                     bool product_reduction) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered = static_cast<int>((index * 17) % 41) - 20;
    const float value = static_cast<float>(centered) / 13.0F;
    result[index] = product_reduction ? 1.0F + value * 0.125F : value;
  }
  return result;
}

struct PreparedBuffers {
  TestTensor output;
  std::vector<std::unique_ptr<hv::DeviceBuffer>> buffers;
  std::vector<flagdnnBinding_t> bindings;
};

inline PreparedBuffers prepare_buffers(std::span<const TestTensor> inputs,
                                       const TestTensor &output,
                                       bool product_reduction,
                                       BindingAddress binding_address,
                                       hv::Stream &stream) {
  PreparedBuffers result;
  result.output = output;
  result.buffers.reserve(inputs.size() + 1);
  result.bindings.reserve(inputs.size() + 1);
  for (const TestTensor &tensor : inputs) {
    const std::vector<float> logical =
        make_input(logical_element_count(tensor), product_reduction);
    const std::vector<std::uint8_t> encoded =
        io::encode(io::scatter(logical, tensor), tensor.data_type);
    auto buffer = std::make_unique<hv::DeviceBuffer>(
        tensor.binding_byte_offset + encoded.size());
    buffer->copy_from_host_at(encoded.data(), encoded.size(),
                              tensor.binding_byte_offset, stream.get());
    void *pointer = binding_pointer(
        buffer->opaque(), tensor.binding_byte_offset, binding_address);
    result.bindings.push_back({tensor.uid, pointer});
    result.buffers.push_back(std::move(buffer));
  }

  const std::vector<float> initial(storage_element_count(output),
                                   io::kPaddingSentinel);
  const std::vector<std::uint8_t> encoded =
      io::encode(initial, output.data_type);
  auto buffer = std::make_unique<hv::DeviceBuffer>(output.binding_byte_offset +
                                                   encoded.size());
  buffer->copy_from_host_at(encoded.data(), encoded.size(),
                            output.binding_byte_offset, stream.get());
  void *pointer = binding_pointer(buffer->opaque(), output.binding_byte_offset,
                                  binding_address);
  result.bindings.push_back({output.uid, pointer});
  result.buffers.push_back(std::move(buffer));
  return result;
}

inline std::vector<float> read_output(const PreparedBuffers &prepared,
                                      hv::Stream &stream,
                                      std::string_view provider) {
  const std::size_t storage = storage_element_count(prepared.output);
  std::vector<std::uint8_t> bytes(
      storage * io::data_type_size(prepared.output.data_type));
  prepared.buffers.back()->copy_to_host_at(bytes.data(), bytes.size(),
                                           prepared.output.binding_byte_offset,
                                           stream.get());
  stream.synchronize();
  const std::vector<float> physical =
      io::decode(bytes, prepared.output.data_type, storage);
  require_padding_unchanged(provider, physical, prepared.output);
  return gather(physical, prepared.output);
}

inline void execute(TestExecutable &executable,
                    std::span<const flagdnnBinding_t> bindings,
                    hv::DeviceBuffer &workspace, hv::Stream &stream) {
  executable.execute(bindings, workspace.opaque(), executable.workspace_size(),
                     stream.opaque());
}

inline void emit_skip(std::string_view operation, std::string_view case_name,
                      std::string_view reason,
                      std::span<const hv::ReferenceTensor> tensors) {
  std::cout << "[SKIP][hipdnn] op=" << operation << " case=" << case_name
            << " reason=" << reason << ' ' << hv::hipdnn_environment() << ' '
            << hv::describe_reference_tensors(tensors) << std::endl;
}

enum class CaseResult { kExecuted, kSkipped };

inline CaseResult
run_layout_case(const LayoutTestCase &test_case,
                const std::function<flagdnn::Handle &()> &get_handle,
                hv::Stream &stream) {
  const hv::HipdnnTensorOperation operation = layout_operation(test_case);
  const std::vector<hv::ReferenceTensor> tensors =
      layout_reference_tensors(test_case);
  const hv::HipdnnCapability capability =
      hv::hipdnn_tensor_capability(operation, tensors);
  hv::require_valid_hipdnn_adapter_contract(
      capability, layout_operation_name(test_case.operation));
  if (!capability.supported) {
    emit_skip(layout_operation_name(test_case.operation), test_case.name,
              capability.reason, tensors);
    return CaseResult::kSkipped;
  }

  const TestTensor reference_output = dense_test_tensor(test_case.output);
  const std::array<TestTensor, 1> inputs = {test_case.input};
  std::unique_ptr<LayoutExecutable> reference;
  PreparedBuffers reference_buffers;
  try {
    reference = build_layout_reference(test_case);
    reference_buffers = prepare_buffers(inputs, reference_output, false,
                                        BindingAddress::kStorageBase, stream);
    hv::DeviceBuffer reference_workspace(reference->workspace_size());
    stream.synchronize();
    execute(*reference, reference_buffers.bindings, reference_workspace,
            stream);
    stream.synchronize();
  } catch (const hv::HipdnnStatusError &error) {
    if (!hv::hipdnn_status_is_capability(error.status())) {
      throw;
    }
    emit_skip(layout_operation_name(test_case.operation), test_case.name,
              std::string("hipDNN tensor runtime capability: ") + error.what(),
              tensors);
    return CaseResult::kSkipped;
  }

  auto flagdnn = build_flagdnn_layout(get_handle(), test_case);
  PreparedBuffers flagdnn_buffers = prepare_buffers(
      inputs, test_case.output, false, BindingAddress::kTensorEntrance, stream);
  hv::DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  stream.synchronize();
  execute(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace, stream);
  stream.synchronize();

  const Accuracy accuracy =
      compare_outputs(read_output(flagdnn_buffers, stream, "FlagDNN"),
                      read_output(reference_buffers, stream, "hipDNN"), 0.0,
                      0.0, test_case.name);
  std::cout << test_case.name
            << ": FlagDNN Graph vs hipDNN primitive PASS max_abs="
            << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
  return CaseResult::kExecuted;
}

inline CaseResult
run_reduction_case(const ReductionTestCase &test_case,
                   const std::function<flagdnn::Handle &()> &get_handle,
                   hv::Stream &stream) {
  const hv::HipdnnTensorOperation operation =
      hv::make_hipdnn_reduction_operation(test_case.mode, test_case.axis,
                                          test_case.keep_dimensions);
  const std::vector<hv::ReferenceTensor> tensors =
      reduction_reference_tensors(test_case);
  const hv::HipdnnCapability capability =
      hv::hipdnn_tensor_capability(operation, tensors);
  hv::require_valid_hipdnn_adapter_contract(capability, "reduction");
  if (!capability.supported) {
    emit_skip("reduction", test_case.name, capability.reason, tensors);
    return CaseResult::kSkipped;
  }

  const TestTensor reference_input =
      reduction_reference_input_tensor(test_case);
  const std::array<TestTensor, 1> reference_inputs = {reference_input};
  const bool product = test_case.mode == FLAGDNN_REDUCTION_MUL;
  std::unique_ptr<ReductionExecutable> reference;
  PreparedBuffers reference_buffers;
  try {
    reference = build_reduction_reference(test_case);
    reference_buffers =
        prepare_buffers(reference_inputs, test_case.output, product,
                        BindingAddress::kStorageBase, stream);
    hv::DeviceBuffer reference_workspace(reference->workspace_size());
    stream.synchronize();
    execute(*reference, reference_buffers.bindings, reference_workspace,
            stream);
    stream.synchronize();
  } catch (const hv::HipdnnStatusError &error) {
    if (!hv::hipdnn_status_is_capability(error.status())) {
      throw;
    }
    emit_skip("reduction", test_case.name,
              std::string("hipDNN tensor runtime capability: ") + error.what(),
              tensors);
    return CaseResult::kSkipped;
  }

  auto flagdnn = build_flagdnn_reduction(get_handle(), test_case);
  const std::array<TestTensor, 1> flagdnn_inputs = {test_case.input};
  PreparedBuffers flagdnn_buffers =
      prepare_buffers(flagdnn_inputs, test_case.output, product,
                      BindingAddress::kTensorEntrance, stream);
  hv::DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  stream.synchronize();
  execute(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace, stream);
  stream.synchronize();

  const Accuracy accuracy =
      compare_outputs(read_output(flagdnn_buffers, stream, "FlagDNN"),
                      read_output(reference_buffers, stream, "hipDNN"),
                      test_case.absolute_tolerance,
                      test_case.relative_tolerance, test_case.name);
  std::cout << test_case.name
            << ": FlagDNN Graph vs hipDNN primitive PASS max_abs="
            << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
  return CaseResult::kExecuted;
}

template <typename Run>
int run_suite(int argc, char **argv, std::string_view family,
              std::string_view suite_name, Run &&run) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY"
              << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    hv::DeviceGuard device;
    hv::Stream stream;
    TemporaryCache cache(family);
    std::unique_ptr<flagdnn::Handle> handle;
    const std::function<flagdnn::Handle &()> get_handle =
        [&]() -> flagdnn::Handle & {
      if (handle == nullptr) {
        handle = std::make_unique<flagdnn::Handle>("hygon", 0);
        handle->set_compiler(argv[1], argv[2], cache.path().string());
      }
      return *handle;
    };
    std::size_t matched = 0;
    std::size_t executed = 0;
    std::size_t skipped = 0;
    run(get_handle, stream, matched, executed, skipped);
    if (matched == 0) {
      throw std::runtime_error("functional case filter matched no cases");
    }
    std::cout << suite_name << ": " << (executed == 0 ? "SKIP" : "PASS")
              << " cases=" << matched << " executed=" << executed
              << " skipped=" << skipped << std::endl;
    return executed == 0 ? kSkipReturnCode : 0;
  } catch (const std::exception &error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << std::endl;
    return 1;
  }
}

} // namespace flagdnn::testing::hygon_functional::tensor

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_TENSOR_RUNNER_SUPPORT_HPP_
