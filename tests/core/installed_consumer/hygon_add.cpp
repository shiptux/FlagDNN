/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <hipdnn.h>
#include <link.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fe = ::flagdnn_frontend;

std::filesystem::path required_environment_path(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    throw std::runtime_error(std::string(name) + " is missing");
  }
  const std::filesystem::path path(value);
  if (!path.is_absolute() || !std::filesystem::exists(path)) {
    throw std::runtime_error(std::string(name) +
                             " is not an existing absolute path");
  }
  return std::filesystem::canonical(path);
}

bool is_within(const std::filesystem::path &root,
               const std::filesystem::path &candidate) {
  const std::filesystem::path relative = candidate.lexically_relative(root);
  return !relative.empty() && !relative.is_absolute() &&
         *relative.begin() != "..";
}

void require_clean_installed_environment() {
  constexpr std::array<const char *, 18> forbidden = {
      "FLAGDNN_BACKEND_ROOT",
      "FLAGDNN_KERNEL_SOURCE_ROOT",
      "FLAGDNN_TUNING_ROOT",
      "FLAGDNN_HYGON_TRITON_JIT_ROOT",
      "FLAGDNN_HYGON_TRITON_JIT_DIR",
      "FLAGDNN_HYGON_TRITON_JIT_LIBRARY",
      "FLAGDNN_HYGON_TRITON_JIT_INCLUDE_DIR",
      "FLAGDNN_HYGON_TRITON_JIT_SCRIPT_DIR",
      "FLAGDNN_HYGON_COMPILER_ENVIRONMENT",
      "FLAGDNN_HYGON_PYTHONPATH",
      "FLAGDNN_TRITON_JIT_LIBRARY",
      "FLAGDNN_TRITON_JIT_INCLUDE_DIR",
      "FLAGDNN_TRITON_JIT_SCRIPT_DIR",
      "FLAGDNN_CODEGEN_COMPILER",
      "FLAGDNN_COMPILER",
      "FLAGDNN_EXECUTION_ENGINE",
      "PYTHONPATH",
      "PYTHONHOME"};
  for (const char *name : forbidden) {
    if (std::getenv(name) != nullptr) {
      throw std::runtime_error(
          std::string("inherited override was not unset: ") + name);
    }
  }
}

void require_installed_jit_resources() {
  const auto sdk = required_environment_path("FLAGDNN_INSTALLED_EXPECTED_SDK");
  const auto expected_library =
      required_environment_path("FLAGDNN_INSTALLED_EXPECTED_JIT");
  const auto expected_scripts =
      required_environment_path("FLAGDNN_INSTALLED_EXPECTED_JIT_SCRIPT_DIR");
  if (!is_within(sdk, expected_library) || !is_within(sdk, expected_scripts)) {
    throw std::runtime_error(
        "installed Hygon JIT resources escape the isolated SDK");
  }

  void *library = dlopen(expected_library.c_str(), RTLD_NOW | RTLD_NOLOAD);
  if (library == nullptr) {
    const char *open_error = dlerror();
    throw std::runtime_error(
        "installed Hygon graph did not map its private libtriton_jit: " +
        std::string(open_error == nullptr ? "unknown dlopen error"
                                          : open_error));
  }
  struct link_map *mapping = nullptr;
  if (dlinfo(library, RTLD_DI_LINKMAP, &mapping) != 0 || mapping == nullptr ||
      mapping->l_name == nullptr || *mapping->l_name == '\0') {
    const char *mapping_error = dlerror();
    (void)dlclose(library);
    throw std::runtime_error(
        "cannot identify the mapped libtriton_jit image: " +
        std::string(mapping_error == nullptr ? "missing link-map path"
                                             : mapping_error));
  }
  const auto mapped_library = std::filesystem::canonical(mapping->l_name);
  if (mapped_library != expected_library) {
    (void)dlclose(library);
    throw std::runtime_error(
        "installed Hygon consumer mapped libtriton_jit outside its SDK "
        "private directory: " +
        mapped_library.string());
  }
  (void)dlclose(library);
}

void check_hip(hipError_t status, std::string_view operation) {
  if (status == hipSuccess) {
    return;
  }
  const char *name = hipGetErrorName(status);
  const char *description = hipGetErrorString(status);
  throw std::runtime_error(
      std::string(operation) + " failed (" +
      (name == nullptr ? "HIP_ERROR_UNKNOWN" : name) + "): " +
      (description == nullptr ? "HIP error description unavailable"
                              : description));
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + status.get_message());
  }
}

void check_hipdnn(hipdnnStatus_t status, std::string_view operation) {
  if (status != HIPDNN_STATUS_SUCCESS) {
    const char *description = hipdnnGetErrorString(status);
    throw std::runtime_error(std::string(operation) + " failed: " +
                             (description == nullptr
                                  ? "hipDNN error description unavailable"
                                  : description));
  }
}

class TemporaryCache {
public:
  TemporaryCache() {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "flagdnn-installed-hygon-add-XXXXXX")
                              .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char *created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
    if (setenv("FLAGDNN_CACHE_DIRECTORY", path_.c_str(), 1) != 0 ||
        setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0 ||
        unsetenv("FLAGDNN_BACKEND") != 0 ||
        unsetenv("FLAGDNN_CODEGEN_COMPILER") != 0) {
      throw std::runtime_error(
          "cannot configure installed Hygon Add environment");
    }
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

class DeviceGuard {
public:
  explicit DeviceGuard(int ordinal) {
    check_hip(hipGetDevice(&previous_), "hipGetDevice");
    check_hip(hipSetDevice(ordinal), "hipSetDevice");
  }

  ~DeviceGuard() { (void)hipSetDevice(previous_); }

  DeviceGuard(const DeviceGuard &) = delete;
  DeviceGuard &operator=(const DeviceGuard &) = delete;

private:
  int previous_ = 0;
};

class Stream {
public:
  Stream() {
    check_hip(hipStreamCreateWithFlags(&value_, hipStreamNonBlocking),
              "hipStreamCreateWithFlags");
  }

  ~Stream() {
    if (value_ != nullptr) {
      (void)hipStreamDestroy(value_);
    }
  }

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  [[nodiscard]] hipStream_t get() const noexcept { return value_; }
  [[nodiscard]] flagdnnStream_t opaque() const noexcept {
    return reinterpret_cast<flagdnnStream_t>(value_);
  }

private:
  hipStream_t value_ = nullptr;
};

class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes == 0 ? 1 : bytes) {
    check_hip(hipMalloc(&value_, bytes_), "hipMalloc");
  }

  ~DeviceBuffer() {
    if (value_ != nullptr) {
      (void)hipFree(value_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  [[nodiscard]] void *get() const noexcept { return value_; }

  void copy_from(const void *source, std::size_t bytes,
                 hipStream_t stream) const {
    if (bytes > bytes_) {
      throw std::invalid_argument("host-to-device copy exceeds buffer");
    }
    check_hip(
        hipMemcpyAsync(value_, source, bytes, hipMemcpyHostToDevice, stream),
        "hipMemcpyAsync(H2D)");
  }

  void copy_to(void *destination, std::size_t bytes, hipStream_t stream) const {
    if (bytes > bytes_) {
      throw std::invalid_argument("device-to-host copy exceeds buffer");
    }
    check_hip(hipMemcpyAsync(destination, value_, bytes, hipMemcpyDeviceToHost,
                             stream),
              "hipMemcpyAsync(D2H)");
  }

private:
  void *value_ = nullptr;
  std::size_t bytes_ = 0;
};

class HipdnnAddReference {
public:
  HipdnnAddReference() {
    check_hipdnn(hipdnnCreate(&handle_), "hipdnnCreate");
    try {
      constexpr int dimensions[] = {1, 1, 256};
      constexpr int strides[] = {256, 256, 1};
      for (auto &descriptor : tensors_) {
        check_hipdnn(hipdnnCreateTensorDescriptor(&descriptor),
                     "hipdnnCreateTensorDescriptor");
        check_hipdnn(hipdnnSetTensorNdDescriptor(descriptor, HIPDNN_DATA_FLOAT,
                                                 3, dimensions, strides),
                     "hipdnnSetTensorNdDescriptor");
      }
      check_hipdnn(hipdnnCreateOpTensorDescriptor(&operation_),
                   "hipdnnCreateOpTensorDescriptor");
      check_hipdnn(hipdnnSetOpTensorDescriptor(operation_, HIPDNN_OP_TENSOR_ADD,
                                               HIPDNN_DATA_FLOAT,
                                               HIPDNN_PROPAGATE_NAN),
                   "hipdnnSetOpTensorDescriptor");
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~HipdnnAddReference() { cleanup(); }

  HipdnnAddReference(const HipdnnAddReference &) = delete;
  HipdnnAddReference &operator=(const HipdnnAddReference &) = delete;

  void execute(hipStream_t stream, void *left, void *right, void *output) {
    check_hipdnn(hipdnnSetStream(handle_, stream), "hipdnnSetStream");
    constexpr float one = 1.0F;
    constexpr float zero = 0.0F;
    check_hipdnn(hipdnnOpTensor(handle_, operation_, &one, tensors_[0], left,
                                &one, tensors_[1], right, &zero, tensors_[2],
                                output),
                 "hipdnnOpTensor(Add)");
  }

private:
  void cleanup() noexcept {
    if (operation_ != nullptr) {
      (void)hipdnnDestroyOpTensorDescriptor(operation_);
      operation_ = nullptr;
    }
    for (auto &descriptor : tensors_) {
      if (descriptor != nullptr) {
        (void)hipdnnDestroyTensorDescriptor(descriptor);
        descriptor = nullptr;
      }
    }
    if (handle_ != nullptr) {
      (void)hipdnnDestroy(handle_);
      handle_ = nullptr;
    }
  }

  hipdnnHandle_t handle_ = nullptr;
  std::array<hipdnnTensorDescriptor_t, 3> tensors_ = {nullptr, nullptr,
                                                      nullptr};
  hipdnnOpTensorDescriptor_t operation_ = nullptr;
};

fe::graph::Graph::Tensor make_tensor(fe::graph::Graph &graph, const char *name,
                                     std::int64_t uid) {
  return graph.tensor(fe::graph::Tensor_attributes()
                          .set_name(name)
                          .set_uid(uid)
                          .set_data_type(fe::DataType_t::FLOAT)
                          .set_dim({1, 1, 256})
                          .set_stride({256, 256, 1}));
}

void require_jit_autotune_artifacts(const std::filesystem::path &cache) {
  std::size_t manifests = 0;
  std::size_t selections = 0;
  std::size_t generated_sources = 0;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(cache)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    manifests += filename == "manifest.json" ? 1U : 0U;
    selections += filename.starts_with(".flagdnn-autotune-v1-stage-") ? 1U : 0U;
    generated_sources += filename.starts_with("generated_stage_") &&
                                 entry.path().extension() == ".py"
                             ? 1U
                             : 0U;
  }
  if (manifests != 1 || selections != 1 || generated_sources != 1) {
    throw std::runtime_error(
        "installed Hygon Add did not use one libtriton_jit artifact, "
        "autotune selection, and generated Triton source");
  }
}

} // namespace

int main() {
  try {
    require_clean_installed_environment();
    TemporaryCache cache;
    DeviceGuard device(0);
    flagdnn::Handle handle;
    Stream stream;

    fe::graph::Graph graph;
    graph.set_name("installed_hygon_add")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(true);
    const auto left = make_tensor(graph, "left", 1);
    const auto right = make_tensor(graph, "right", 2);
    auto output =
        graph.pointwise(left, right,
                        fe::graph::Pointwise_attributes()
                            .set_name("add")
                            .set_mode(fe::PointwiseMode_t::ADD)
                            .set_compute_data_type(fe::DataType_t::FLOAT));
    output->set_name("output")
        .set_uid(3)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({1, 1, 256})
        .set_stride({256, 256, 1})
        .set_output(true);
    check_frontend(graph.build(handle, {fe::HeurMode_t::A}),
                   "installed Hygon Add graph build");
    require_installed_jit_resources();
    require_jit_autotune_artifacts(cache.path());

    std::array<float, 256> host_left{};
    std::array<float, 256> host_right{};
    std::array<float, 256> host_output{};
    std::array<float, 256> hipdnn_output{};
    for (std::size_t index = 0; index < host_left.size(); ++index) {
      host_left[index] = static_cast<float>(index) * 0.25F;
      host_right[index] = 7.0F - static_cast<float>(index) * 0.125F;
    }
    constexpr std::size_t tensor_bytes = sizeof(host_left);
    DeviceBuffer device_left(tensor_bytes);
    DeviceBuffer device_right(tensor_bytes);
    DeviceBuffer device_output(tensor_bytes);
    DeviceBuffer hipdnn_device_output(tensor_bytes);
    const std::int64_t workspace_size = graph.get_workspace_size();
    if (workspace_size < 0) {
      throw std::runtime_error(
          "installed Hygon Add returned negative workspace");
    }
    DeviceBuffer workspace(static_cast<std::size_t>(workspace_size));

    device_left.copy_from(host_left.data(), tensor_bytes, stream.get());
    device_right.copy_from(host_right.data(), tensor_bytes, stream.get());
    HipdnnAddReference reference;
    reference.execute(stream.get(), device_left.get(), device_right.get(),
                      hipdnn_device_output.get());
    const std::array<flagdnnBinding_t, 3> bindings = {
        flagdnnBinding_t{1, device_left.get()},
        flagdnnBinding_t{2, device_right.get()},
        flagdnnBinding_t{3, device_output.get()}};
    check_frontend(graph.execute(handle, bindings, workspace.get(),
                                 static_cast<std::size_t>(workspace_size),
                                 stream.opaque()),
                   "installed Hygon Add execute");
    device_output.copy_to(host_output.data(), tensor_bytes, stream.get());
    hipdnn_device_output.copy_to(hipdnn_output.data(), tensor_bytes,
                                 stream.get());
    check_hip(hipStreamSynchronize(stream.get()), "hipStreamSynchronize");

    for (std::size_t index = 0; index < host_output.size(); ++index) {
      if (std::abs(host_output[index] - hipdnn_output[index]) > 1.0e-6F) {
        throw std::runtime_error(
            "installed Hygon Add differs from hipDNN at index " +
            std::to_string(index));
      }
    }

    std::cout << "PASS installed FlagDNN C++ Graph Add -> libtriton_jit -> "
                 "autotune -> Hygon GPU; reference=hipDNN OpTensor Add\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
