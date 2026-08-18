/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/libtriton_jit.hpp"

#include "backends/ascend/engines/batchnorm_oracle.hpp"
#include "backends/ascend/engines/batchnorm_inference_oracle.hpp"
#include "backends/ascend/engines/build_time_prewarm.hpp"
#include "backends/ascend/engines/convolution_fprop_oracle.hpp"
#include "backends/ascend/engines/layernorm_oracle.hpp"
#include "backends/ascend/engines/matmul_oracle.hpp"
#include "backends/ascend/engines/python_stdout_containment.hpp"
#include "backends/ascend/engines/reduction_oracle.hpp"
#include "backends/ascend/engines/rmsnorm_oracle.hpp"

#include "backends/ascend/error.hpp"
#include "src/runtime/json.hpp"
#include "src/runtime/sha256.hpp"

#include <Python.h>
#include <triton_jit/kernel_metadata.h>
#include <triton_jit/triton_jit_function.h>
#include <triton_jit/triton_kernel.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

/* libtriton_jit explicitly instantiates and exports its NPU function class.
 * Suppress this plugin TU's implicit instantiation of the inline definition;
 * otherwise NpuBackend::ensure_context() is copied into the plugin and creates
 * forbidden direct aclrtSetDevice/aclrtCreateContext references even though the
 * outer ContextGuard guarantees that branch is never taken. */
namespace triton_jit {
extern template class TritonKernelImpl<NpuBackend>;
extern template class TritonJITFunctionImpl<NpuBackend>;
}  // namespace triton_jit

namespace flagdnn::ascend {
namespace {

namespace fs = std::filesystem;
using JsonValue = flagdnn::native::json::Value;
using LtjFunction = triton_jit::TritonJITFunction;

constexpr std::size_t kMaximumMetadataBytes = 1U << 20U;
constexpr std::size_t kMaximumCacheFiles = 4096;
constexpr std::uintmax_t kMaximumCacheBytes = 1ULL << 30U;
constexpr std::size_t kGraphWorkspaceAlignment = 256;

using LtjRawLaunchMethod = void (LtjFunction::*)(
    triton_jit::NpuBackend::StreamType,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    std::string,
    void**,
    std::size_t) const;

[[nodiscard]] LtjRawLaunchMethod exported_raw_launch_method() noexcept {
  /* The installed LTJ explicitly instantiates and exports this member.  Keep
   * the typed pointer volatile so this TU emits an indirect call to that
   * exported instantiation instead of inlining NpuBackend::ensure_context().
   * Inlining the header implementation would give this caller-owned plugin
   * forbidden direct aclrtSetDevice/aclrtCreateContext references. */
  static const volatile LtjRawLaunchMethod method =
      &LtjFunction::launch_with_raw_args;
  return method;
}

void launch_with_exported_raw_api(LtjFunction& function,
                                  aclrtStream stream,
                                  const LtjNpuRawCandidate& candidate,
                                  void** arguments,
                                  std::size_t argument_count) {
  const LtjRawLaunchMethod method = exported_raw_launch_method();
  (function.*method)(stream,
                     candidate.grid[0],
                     candidate.grid[1],
                     candidate.grid[2],
                     candidate.num_warps,
                     candidate.num_stages,
                     candidate.full_signature,
                     arguments,
                     argument_count);
}

struct ContainedRawLaunch {
  LtjFunction* function = nullptr;
  aclrtStream stream = nullptr;
  const LtjNpuRawCandidate* candidate = nullptr;
  void** arguments = nullptr;
  std::size_t argument_count = 0;
  bool* raw_started = nullptr;
};

void launch_create_raw_with_contained_stdout(void* opaque) {
  auto* launch = static_cast<ContainedRawLaunch*>(opaque);
  if (launch == nullptr || launch->function == nullptr ||
      launch->candidate == nullptr || launch->raw_started == nullptr) {
    throw std::invalid_argument("invalid contained Ascend raw launch");
  }
  *launch->raw_started = true;
  launch_with_exported_raw_api(*launch->function,
                               launch->stream,
                               *launch->candidate,
                               launch->arguments,
                               launch->argument_count);
}

[[noreturn]] void compilation_failure(std::string message) {
  throw AscendError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    std::move(message));
}

[[nodiscard]] bool same_time(const timespec& left,
                             const timespec& right) noexcept {
  return left.tv_sec == right.tv_sec && left.tv_nsec == right.tv_nsec;
}

class FileDescriptor {
 public:
  explicit FileDescriptor(int value) noexcept : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0) {
      (void)::close(value_);
    }
  }

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  [[nodiscard]] int get() const noexcept { return value_; }

 private:
  int value_ = -1;
};

struct RegularFile {
  std::string contents;
  std::string sha256;
  std::uint64_t device = 0;
  std::uint64_t inode = 0;
  std::uintmax_t size = 0;
  timespec modification_time{};
  timespec change_time{};
};

[[nodiscard]] RegularFile read_regular_file(const fs::path& path,
                                            std::size_t maximum_size,
                                            bool retain_contents = true) {
  const int descriptor =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    compilation_failure("cannot open Ascend cache file without following links: " +
                        path.string());
  }
  FileDescriptor owner(descriptor);
  struct stat before {};
  if (::fstat(owner.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_nlink != 1 || before.st_size < 0 ||
      static_cast<std::uintmax_t>(before.st_size) > maximum_size) {
    compilation_failure("Ascend cache file is not a bounded private regular file: " +
                        path.string());
  }

  RegularFile result;
  result.contents.resize(static_cast<std::size_t>(before.st_size));
  std::size_t offset = 0;
  while (offset < result.contents.size()) {
    const ssize_t count = ::read(owner.get(), result.contents.data() + offset,
                                 result.contents.size() - offset);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      compilation_failure("cannot read complete Ascend cache file: " +
                          path.string());
    }
    offset += static_cast<std::size_t>(count);
  }

  struct stat after {};
  if (::fstat(owner.get(), &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_size != after.st_size ||
      !same_time(before.st_mtim, after.st_mtim) ||
      !same_time(before.st_ctim, after.st_ctim)) {
    compilation_failure("Ascend cache file changed while it was inspected: " +
                        path.string());
  }
  result.sha256 = flagdnn::native::sha256(result.contents);
  if (!retain_contents) {
    std::string{}.swap(result.contents);
  }
  result.device = static_cast<std::uint64_t>(after.st_dev);
  result.inode = static_cast<std::uint64_t>(after.st_ino);
  result.size = static_cast<std::uintmax_t>(after.st_size);
  result.modification_time = after.st_mtim;
  result.change_time = after.st_ctim;
  return result;
}

[[nodiscard]] bool same_file(const RegularFile& left,
                             const RegularFile& right) noexcept {
  return left.device == right.device && left.inode == right.inode &&
         left.size == right.size && left.sha256 == right.sha256 &&
         same_time(left.modification_time, right.modification_time) &&
         same_time(left.change_time, right.change_time);
}

[[nodiscard]] bool same_file_contents(const RegularFile& left,
                                      const RegularFile& right) noexcept {
  return left.size == right.size && left.sha256 == right.sha256;
}

using CacheSnapshot = std::map<fs::path, RegularFile>;

[[nodiscard]] fs::path validate_private_cache_root(const fs::path& configured) {
  if (configured.empty() || !configured.is_absolute()) {
    compilation_failure("development Ascend cache root must be absolute");
  }
  std::error_code error;
  const fs::path canonical = fs::canonical(configured, error);
  if (error || canonical != configured.lexically_normal()) {
    compilation_failure("development Ascend cache root must exist and be canonical");
  }
  struct stat status {};
  if (::lstat(canonical.c_str(), &status) != 0 ||
      !S_ISDIR(status.st_mode) || S_ISLNK(status.st_mode) ||
      status.st_uid != ::geteuid() || (status.st_mode & 0077) != 0) {
    compilation_failure(
        "development Ascend cache root must be a private owner-only directory");
  }
  const char* environment = std::getenv("TRITON_CACHE_DIR");
  if (environment == nullptr || fs::path(environment) != canonical) {
    compilation_failure(
        "TRITON_CACHE_DIR differs from the process-bound Ascend cache root");
  }
  return canonical;
}

[[nodiscard]] CacheSnapshot scan_cache(const fs::path& root,
                                       std::string_view entry_point) {
  const std::string filename = std::string(entry_point) + ".json";
  CacheSnapshot result;
  std::size_t file_count = 0;
  std::uintmax_t total_bytes = 0;
  std::error_code error;
  fs::recursive_directory_iterator iterator(
      root, fs::directory_options::none, error);
  const fs::recursive_directory_iterator end;
  if (error) {
    compilation_failure("cannot enter the private Ascend cache root");
  }
  while (iterator != end) {
    const fs::directory_entry entry = *iterator;
    const fs::file_status status = entry.symlink_status(error);
    if (error) {
      compilation_failure("cannot stat an Ascend cache entry");
    }
    if (fs::is_symlink(status) ||
        (!fs::is_directory(status) && !fs::is_regular_file(status))) {
      compilation_failure(
          "private Ascend cache contains a link or non-regular entry");
    }
    if (fs::is_regular_file(status)) {
      if (++file_count > kMaximumCacheFiles ||
          total_bytes > kMaximumCacheBytes) {
        compilation_failure("private Ascend cache exceeds its scan budget");
      }
      const std::uintmax_t remaining = kMaximumCacheBytes - total_bytes;
      const bool is_target_metadata = entry.path().filename() == filename;
      const std::uintmax_t per_file_limit =
          is_target_metadata
              ? std::min<std::uintmax_t>(remaining, kMaximumMetadataBytes)
              : remaining;
      RegularFile file = read_regular_file(
          entry.path(), static_cast<std::size_t>(per_file_limit),
          is_target_metadata);
      if (file.size > remaining) {
        compilation_failure("private Ascend cache exceeds its scan budget");
      }
      total_bytes += file.size;
      const auto [unused, inserted] = result.emplace(
          entry.path().lexically_normal(), std::move(file));
      (void)unused;
      if (!inserted) {
        compilation_failure("private Ascend cache contains a duplicate path");
      }
    }
    iterator.increment(error);
    if (error) {
      compilation_failure("cannot complete the private Ascend cache scan");
    }
  }
  return result;
}

[[nodiscard]] bool same_cache_snapshot(const CacheSnapshot& left,
                                       const CacheSnapshot& right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  for (const auto& [path, state] : left) {
    const auto found = right.find(path);
    if (found == right.end() || !same_file(state, found->second)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] const JsonValue* member(const JsonValue::Object& object,
                                      std::string_view name) noexcept {
  const auto found = object.find(name);
  return found == object.end() ? nullptr : &found->second;
}

[[nodiscard]] RawArgumentType metadata_argument_type(
    const JsonValue& value) {
  const auto& object = value.as_object();
  const JsonValue* encoded = member(object, "type");
  if (encoded == nullptr) {
    compilation_failure("NPU metadata argument has no type");
  }
  const std::string& type = encoded->as_string();
  if (type == "ptr" || type == "pointer") {
    return RawArgumentType::kPointer;
  }
  if (type == "i32" || type == "u32") {
    return RawArgumentType::kI32;
  }
  if (type == "i64" || type == "u64") {
    return RawArgumentType::kI64;
  }
  if (type == "fp32" || type == "f32") {
    return RawArgumentType::kF32;
  }
  if (type == "fp64" || type == "f64") {
    return RawArgumentType::kF64;
  }
  compilation_failure("NPU metadata contains an unsupported runtime type");
}

[[nodiscard]] triton_jit::NpuArgType ltj_argument_type(
    RawArgumentType type) {
  switch (type) {
    case RawArgumentType::kPointer:
      return triton_jit::NpuArgType::POINTER;
    case RawArgumentType::kI32:
      return triton_jit::NpuArgType::I32;
    case RawArgumentType::kI64:
      return triton_jit::NpuArgType::I64;
    case RawArgumentType::kF32:
      return triton_jit::NpuArgType::F32;
    case RawArgumentType::kF64:
      return triton_jit::NpuArgType::F64;
  }
  compilation_failure("unknown Ascend raw argument type");
}

void validate_metadata(const fs::path& path,
                       const RegularFile& file,
                       const fs::path& cache_root,
                       const CacheSnapshot& snapshot,
                       const LtjNpuRawCandidate& candidate,
                       unsigned int observed_shared_memory) {
  try {
    const JsonValue document = flagdnn::native::json::parse(file.contents);
    const auto& object = document.as_object();
    const JsonValue* layout = member(object, "arg_layout");
    if (layout == nullptr) {
      compilation_failure("NPU metadata is missing its runtime argument layout");
    }

    std::uint64_t workspace_size = 0;
    if (const JsonValue* workspace = member(object, "workspace_size")) {
      const std::int64_t encoded = workspace->as_int();
      if (encoded < 0) {
        compilation_failure("NPU metadata workspace size is negative");
      }
      workspace_size = static_cast<std::uint64_t>(encoded);
    }
    if (workspace_size != 0) {
      compilation_failure(
          "development Ascend pointwise requires zero libtriton_jit workspace");
    }

    unsigned int shared_memory = 0;
    if (const JsonValue* shared = member(object, "shared")) {
      const std::int64_t encoded = shared->as_int();
      if (encoded < 0 ||
          static_cast<std::uint64_t>(encoded) >
              std::numeric_limits<unsigned int>::max()) {
        compilation_failure("NPU metadata shared memory is out of range");
      }
      shared_memory = static_cast<unsigned int>(encoded);
    }
    if (shared_memory != observed_shared_memory) {
      compilation_failure(
          "NPU metadata shared memory differs from launch-enter metadata");
    }

    const auto& arguments = layout->as_array();
    if (arguments.size() != candidate.argument_types.size()) {
      compilation_failure("NPU metadata runtime ABI count differs");
    }
    for (std::size_t index = 0; index < arguments.size(); ++index) {
      if (metadata_argument_type(arguments[index]) !=
          candidate.argument_types[index]) {
        compilation_failure("NPU metadata runtime ABI type differs");
      }
    }

    const fs::path binary = path.parent_path() /
                            (candidate.entry_point + std::string(".npubin"));
    const auto binary_state = snapshot.find(binary.lexically_normal());
    if (binary_state == snapshot.end() || binary_state->second.size == 0 ||
        binary_state->second.sha256.empty()) {
      compilation_failure(
          "NPU metadata has no matching no-follow attested npubin");
    }

    const triton_jit::NpuKernelMetadata normalized =
        triton_jit::load_npu_metadata(path.parent_path().string(),
                                      candidate.entry_point);
    if (normalized.workspace_size != 0 ||
        normalized.shared != observed_shared_memory ||
        normalized.arg_layout.size() != candidate.argument_types.size()) {
      compilation_failure("public NPU metadata loader disagrees with cache proof");
    }
    for (std::size_t index = 0; index < normalized.arg_layout.size(); ++index) {
      if (normalized.arg_layout[index].type !=
          ltj_argument_type(candidate.argument_types[index])) {
        compilation_failure("public NPU metadata ABI disagrees with artifact");
      }
    }
    const CacheSnapshot unchanged = scan_cache(cache_root, candidate.entry_point);
    if (!same_cache_snapshot(snapshot, unchanged)) {
      compilation_failure(
          "public NPU metadata loading changed the private cache tree");
    }
  } catch (const AscendError&) {
    throw;
  } catch (const std::exception& error) {
    compilation_failure("cannot validate NPU metadata " + path.string() +
                        ": " + error.what());
  }
}

[[nodiscard]] std::string candidate_key(const fs::path& cache_root,
                                        const EngineBuildContext& context,
                                        const LtjNpuRawCandidate& candidate) {
  std::ostringstream output;
  output << cache_root.string() << '\n' << context.configuration_identity << '\n'
         << candidate.source.string() << '\n' << candidate.source_sha256 << '\n'
         << candidate.entry_point << '\n' << candidate.full_signature << '\n'
         << context.device_ordinal << '\n' << candidate.num_warps << '\n'
         << candidate.num_stages;
  return output.str();
}

struct CacheAttestation {
  fs::path metadata_path;
  CacheSnapshot published_files;
};

[[nodiscard]] std::map<std::string, CacheAttestation>& cache_attestations() {
  static std::map<std::string, CacheAttestation> attestations;
  return attestations;
}

[[nodiscard]] const RegularFile& select_metadata(
    const CacheSnapshot& before,
    const CacheSnapshot& after,
    const std::string& key,
    std::string_view entry_point,
    fs::path* selected_path) {
  auto& attestations = cache_attestations();
  const auto known = attestations.find(key);
  if (known != attestations.end()) {
    if (!same_cache_snapshot(before, after)) {
      compilation_failure(
          "prewarmed raw launch changed the private NPU cache tree");
    }
    for (const auto& [path, state] : known->second.published_files) {
      const auto current = before.find(path);
      if (current == before.end() ||
          !same_file_contents(state, current->second)) {
        compilation_failure(
            "an attested NPU cache file changed between raw launches");
      }
    }
    const auto current = after.find(known->second.metadata_path);
    if (current == after.end()) {
      compilation_failure("prewarmed NPU metadata disappeared");
    }
    *selected_path = current->first;
    return current->second;
  }

  CacheSnapshot published_files;
  for (const auto& [path, state] : before) {
    const auto current = after.find(path);
    if (current == after.end()) {
      compilation_failure(
          "first raw compilation removed existing NPU cache file: " +
          path.string());
    }
    if (!same_file(state, current->second)) {
      if (!same_file_contents(state, current->second)) {
        compilation_failure(
            "first raw compilation changed existing NPU cache contents: " +
            path.string());
      }
      RegularFile refreshed = current->second;
      std::string{}.swap(refreshed.contents);
      published_files.emplace(path, std::move(refreshed));
    }
  }

  const std::string metadata_filename = std::string(entry_point) + ".json";
  std::size_t target_metadata_count = 0;
  for (const auto& [path, state] : after) {
    const auto previous = before.find(path);
    if (previous == before.end()) {
      RegularFile published = state;
      std::string{}.swap(published.contents);
      published_files.emplace(path, std::move(published));
    }
    if (published_files.find(path) != published_files.end() &&
        path.filename() == metadata_filename) {
      ++target_metadata_count;
      *selected_path = path;
    }
  }
  if (target_metadata_count != 1) {
    compilation_failure(
        "first raw compilation must publish or identity-refresh exactly one "
        "target entry metadata file");
  }
  const fs::path binary = selected_path->parent_path() /
                          (std::string(entry_point) + ".npubin");
  const auto binary_state = after.find(binary.lexically_normal());
  if (binary_state == after.end() || binary_state->second.size == 0 ||
      binary_state->second.sha256.empty()) {
    compilation_failure(
        "first raw compilation has no matching attested npubin");
  }
  RegularFile published_binary = binary_state->second;
  std::string{}.swap(published_binary.contents);
  published_files.insert_or_assign(binary.lexically_normal(),
                                   std::move(published_binary));
  const auto selected = after.find(*selected_path);
  if (selected == after.end()) {
    compilation_failure("selected NPU metadata is absent from the cache snapshot");
  }
  const auto [attestation, inserted] = attestations.emplace(
      key, CacheAttestation{*selected_path, std::move(published_files)});
  (void)attestation;
  if (!inserted) {
    compilation_failure("NPU cache attestation key was inserted concurrently");
  }
  return selected->second;
}

void validate_source(const LtjNpuRawCandidate& candidate) {
  std::error_code error;
  const fs::path canonical = fs::canonical(candidate.source, error);
  const fs::file_status status = fs::symlink_status(candidate.source, error);
  if (error || canonical != candidate.source.lexically_normal() ||
      !fs::is_regular_file(status) || fs::is_symlink(status)) {
    compilation_failure("materialized Ascend source is not canonical and regular");
  }
  const RegularFile source =
      read_regular_file(candidate.source, kMaximumMetadataBytes);
  if (source.sha256 != candidate.source_sha256) {
    compilation_failure("materialized Ascend source hash changed before JIT");
  }
}

void* find_binding(const flagdnnBackendBindingV2 bindings[],
                   std::size_t binding_count,
                   std::int64_t uid) {
  void* result = nullptr;
  std::size_t matches = 0;
  for (std::size_t index = 0; index < binding_count; ++index) {
    if (bindings[index].uid == uid) {
      result = bindings[index].device_pointer;
      ++matches;
    }
  }
  require(matches == 1 && result != nullptr,
          "a required Ascend binding is missing or duplicated");
  return result;
}

void validate_execution_inputs(const AscendArtifact& artifact,
                               const flagdnnBackendBindingV2 bindings[],
                               std::size_t binding_count,
                               void* workspace,
                               std::size_t workspace_size) {
  require(binding_count == artifact.binding_uids.size() &&
              (binding_count == 0 || bindings != nullptr),
          "Ascend binding count does not match the executable");
  for (const std::int64_t uid : artifact.binding_uids) {
    (void)find_binding(bindings, binding_count, uid);
  }
  require(workspace_size >= artifact.workspace_size,
          "Ascend Graph workspace is smaller than the executable requirement");
  if (artifact.workspace_size != 0) {
    require(workspace != nullptr &&
                reinterpret_cast<std::uintptr_t>(workspace) %
                        kGraphWorkspaceAlignment ==
                    0,
            "Ascend Graph workspace must be non-null and 256-byte aligned");
  }
}

[[nodiscard]] std::size_t storage_element_size(StorageDataType type) noexcept {
  switch (type) {
    case StorageDataType::kFloat32:
      return 4U;
    case StorageDataType::kFloat16:
    case StorageDataType::kBFloat16:
      return 2U;
    case StorageDataType::kBoolean:
      return 1U;
  }
  return 0U;
}

[[nodiscard]] std::size_t kernel_input_count(KernelFamily family) noexcept {
  switch (family) {
    case KernelFamily::kUnary:
    case KernelFamily::kLayout:
    case KernelFamily::kReduction:
      return 1U;
    case KernelFamily::kBinary:
      return 2U;
    case KernelFamily::kMatMul:
      return matmul_kernel_input_count();
    case KernelFamily::kConvolutionFprop:
      return convolution_fprop_kernel_input_count();
    case KernelFamily::kTernary:
      return 3U;
    case KernelFamily::kBatchNorm:
      return batchnorm_kernel_input_count();
    case KernelFamily::kBatchNormInference:
      return batchnorm_inference_kernel_input_count();
    case KernelFamily::kRmsNorm:
      return rmsnorm_kernel_input_count();
    case KernelFamily::kLayerNorm:
      return layernorm_kernel_input_count();
  }
  return 0U;
}

[[nodiscard]] std::size_t kernel_output_count(KernelFamily family) noexcept {
  return family == KernelFamily::kBatchNorm
             ? batchnorm_tensor_slot_count() - batchnorm_kernel_input_count()
             : family == KernelFamily::kLayerNorm
             ? 3U
             : family == KernelFamily::kRmsNorm ? 2U : 1U;
}

[[nodiscard]] std::size_t kernel_runtime_argument_count(
    KernelFamily family) noexcept {
  return kernel_input_count(family) + kernel_output_count(family) + 1U;
}

[[nodiscard]] float decode_storage_value(const std::uint8_t* bytes,
                                         StorageDataType type) noexcept {
  if (type == StorageDataType::kBoolean) {
    return bytes[0] == 0U ? 0.0F : 1.0F;
  }
  if (type == StorageDataType::kFloat32) {
    float result = 0.0F;
    std::memcpy(&result, bytes, sizeof(result));
    return result;
  }
  std::uint16_t encoded = 0;
  std::memcpy(&encoded, bytes, sizeof(encoded));
  if (type == StorageDataType::kBFloat16) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(encoded) << 16U);
  }

  const std::uint32_t sign =
      static_cast<std::uint32_t>(encoded & 0x8000U) << 16U;
  std::uint32_t exponent = (encoded >> 10U) & 0x1FU;
  std::uint32_t mantissa = encoded & 0x03FFU;
  std::uint32_t bits = sign;
  if (exponent == 0) {
    if (mantissa != 0) {
      exponent = 113U;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1U;
        --exponent;
      }
      mantissa &= 0x03FFU;
      bits |= exponent << 23U;
      bits |= mantissa << 13U;
    }
  } else if (exponent == 0x1FU) {
    bits |= 0x7F800000U | (mantissa << 13U);
  } else {
    bits |= (exponent + 112U) << 23U;
    bits |= mantissa << 13U;
  }
  return std::bit_cast<float>(bits);
}

void fill_storage_pattern(std::vector<std::uint8_t>& bytes,
                          StorageDataType type,
                          std::int64_t uid,
                          bool positive_only,
                          bool sigmoid_backward_logit,
                          int comparison_operand) {
  const std::size_t element_size = storage_element_size(type);
  require(bytes.size() % element_size == 0,
          "Ascend build-time input storage has a partial element",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  for (std::size_t offset = 0; offset < bytes.size();
       offset += element_size) {
    const std::size_t element = offset / element_size;
    const std::size_t pattern = comparison_operand >= 0
                                    ? element % 4U
                                    : (element + static_cast<std::size_t>(uid)) %
                                          4U;
    std::array<std::uint8_t, 4> encoded{};
    if (type == StorageDataType::kBoolean) {
      static constexpr std::array<std::uint8_t, 4> kBooleanValues = {
          0U, 1U, 1U, 0U};
      encoded[0] = kBooleanValues[pattern];
    } else if (type == StorageDataType::kFloat32) {
      static constexpr std::array<float, 4> kSignedValues = {
          1.0F, 2.0F, -1.0F, -2.0F};
      static constexpr std::array<float, 4> kPositiveValues = {
          0.5F, 1.0F, 2.0F, 4.0F};
      static constexpr std::array<float, 4> kSigmoidBackwardLogits = {
          -10.0F, 10.0F, -20.0F, 20.0F};
      static constexpr std::array<float, 4> kComparisonLeft = {
          -1.0F, 0.0F, 2.0F, 4.0F};
      static constexpr std::array<float, 4> kComparisonRight = {
          -1.0F, 1.0F, 1.0F, 4.0F};
      const auto& values = comparison_operand == 0
                               ? kComparisonLeft
                               : comparison_operand == 1
                                     ? kComparisonRight
                                     : sigmoid_backward_logit
                                           ? kSigmoidBackwardLogits
                                           : positive_only ? kPositiveValues
                                                           : kSignedValues;
      std::memcpy(encoded.data(), &values[pattern], sizeof(float));
    } else {
      static constexpr std::array<std::uint16_t, 4> kSignedFloat16Values = {
          0x3C00U, 0x4000U, 0xBC00U, 0xC000U};
      static constexpr std::array<std::uint16_t, 4> kPositiveFloat16Values = {
          0x3800U, 0x3C00U, 0x4000U, 0x4400U};
      static constexpr std::array<std::uint16_t, 4> kSignedBFloat16Values = {
          0x3F80U, 0x4000U, 0xBF80U, 0xC000U};
      static constexpr std::array<std::uint16_t, 4> kPositiveBFloat16Values = {
          0x3F00U, 0x3F80U, 0x4000U, 0x4080U};
      static constexpr std::array<std::uint16_t, 4>
          kSigmoidBackwardFloat16Logits = {
              0xC900U, 0x4900U, 0xCD00U, 0x4D00U};
      static constexpr std::array<std::uint16_t, 4>
          kSigmoidBackwardBFloat16Logits = {
              0xC120U, 0x4120U, 0xC1A0U, 0x41A0U};
      static constexpr std::array<std::uint16_t, 4> kComparisonFloat16Left = {
          0xBC00U, 0x0000U, 0x4000U, 0x4400U};
      static constexpr std::array<std::uint16_t, 4> kComparisonFloat16Right = {
          0xBC00U, 0x3C00U, 0x3C00U, 0x4400U};
      static constexpr std::array<std::uint16_t, 4> kComparisonBFloat16Left = {
          0xBF80U, 0x0000U, 0x4000U, 0x4080U};
      static constexpr std::array<std::uint16_t, 4> kComparisonBFloat16Right = {
          0xBF80U, 0x3F80U, 0x3F80U, 0x4080U};
      const std::uint16_t encoded_value =
          comparison_operand == 0
              ? (type == StorageDataType::kFloat16
                     ? kComparisonFloat16Left[pattern]
                     : kComparisonBFloat16Left[pattern])
              : comparison_operand == 1
                    ? (type == StorageDataType::kFloat16
                           ? kComparisonFloat16Right[pattern]
                           : kComparisonBFloat16Right[pattern])
                    : sigmoid_backward_logit
              ? (type == StorageDataType::kFloat16
                     ? kSigmoidBackwardFloat16Logits[pattern]
                     : kSigmoidBackwardBFloat16Logits[pattern])
              : type == StorageDataType::kFloat16
                    ? (positive_only ? kPositiveFloat16Values[pattern]
                                     : kSignedFloat16Values[pattern])
                    : (positive_only ? kPositiveBFloat16Values[pattern]
                                     : kSignedBFloat16Values[pattern]);
      std::memcpy(encoded.data(), &encoded_value, sizeof(encoded_value));
    }
    std::memcpy(bytes.data() + offset, encoded.data(), element_size);
  }
}

class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  explicit DeviceAllocation(std::size_t size) : size_(size) {
    if (size_ != 0) {
      check_acl(aclrtMalloc(&value_, size_, ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(development prewarm)");
    }
  }
  ~DeviceAllocation() {
    if (!release_noexcept()) {
      latch_process_terminal();
    }
  }

  DeviceAllocation(DeviceAllocation&& other) noexcept
      : value_(std::exchange(other.value_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}
  DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
    if (this != &other) {
      if (!release_noexcept()) {
        latch_process_terminal();
      }
      value_ = std::exchange(other.value_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  [[nodiscard]] void* get() const noexcept { return value_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  void release() {
    if (value_ == nullptr) {
      return;
    }
    void* value = std::exchange(value_, nullptr);
    size_ = 0;
    check_acl(aclrtFree(value), "aclrtFree(development prewarm)");
  }

  [[nodiscard]] bool release_noexcept() noexcept {
    if (value_ == nullptr) {
      return true;
    }
    void* value = std::exchange(value_, nullptr);
    size_ = 0;
    return aclrtFree(value) == ACL_SUCCESS;
  }

  void abandon() noexcept {
    value_ = nullptr;
    size_ = 0;
  }

 private:
  void* value_ = nullptr;
  std::size_t size_ = 0;
};

class BuildResources {
 public:
  BuildResources() = default;
  ~BuildResources() {
    if (!release_noexcept()) {
      latch_process_terminal();
    }
  }

  BuildResources(const BuildResources&) = delete;
  BuildResources& operator=(const BuildResources&) = delete;

  void initialize(const AscendArtifact& artifact) {
    require(stream_ == nullptr && allocations_.empty() && bindings_.empty() &&
                binding_shadows_.empty() && workspace_.get() == nullptr &&
                workspace_shadow_.empty() && reduction_input_shadow_.empty(),
            "Ascend development prewarm resources were initialized twice",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
    check_acl(aclrtCreateStream(&stream_), "aclrtCreateStream");
    std::map<std::int64_t, std::pair<std::size_t, std::size_t>> requirements;
    for (const AscendStageArtifact& stage : artifact.stages) {
      for (const ArgumentSource& argument : stage.arguments) {
        if (argument.source == ArgumentSourceKind::kBinding) {
          auto& requirement = requirements[argument.uid];
          requirement.first = std::max(requirement.first, argument.size);
          requirement.second = std::max(requirement.second, argument.alignment);
        }
      }
    }
    allocations_.reserve(artifact.binding_uids.size());
    bindings_.reserve(artifact.binding_uids.size());
    binding_shadows_.reserve(artifact.binding_uids.size());
    for (const std::int64_t uid : artifact.binding_uids) {
      const auto found = requirements.find(uid);
      require(found != requirements.end() && found->second.first != 0,
              "Ascend prewarm binding has no allocation description",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      allocations_.emplace_back(found->second.first);
      binding_shadows_.emplace_back(found->second.first, 0U);
      void* pointer = allocations_.back().get();
      require(pointer != nullptr &&
                  reinterpret_cast<std::uintptr_t>(pointer) %
                          found->second.second ==
                      0,
              "Ascend prewarm allocation does not satisfy artifact alignment",
              FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
      bindings_.push_back({uid, pointer});
      check_acl(aclrtMemsetAsync(pointer,
                                 allocations_.back().size(),
                                 0,
                                 allocations_.back().size(),
                                 stream_),
                "aclrtMemsetAsync(development binding)");
      synchronized_ = false;
    }
    if (artifact.workspace_size != 0) {
      workspace_ = DeviceAllocation(artifact.workspace_size);
      workspace_shadow_.assign(artifact.workspace_size, 0U);
      require(reinterpret_cast<std::uintptr_t>(workspace_.get()) %
                      kGraphWorkspaceAlignment ==
                  0,
              "Ascend prewarm Graph workspace is not 256-byte aligned",
              FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
      check_acl(aclrtMemsetAsync(workspace_.get(),
                                 workspace_.size(),
                                 0,
                                 workspace_.size(),
                                 stream_),
                "aclrtMemsetAsync(development workspace)");
      synchronized_ = false;
    }
    synchronize();
    seed_external_inputs(artifact);
  }

  void synchronize() {
    check_acl(aclrtSynchronizeStream(stream_),
              "aclrtSynchronizeStream(development prewarm)");
    synchronized_ = true;
  }

  [[nodiscard]] bool synchronize_noexcept() noexcept {
    if (stream_ != nullptr && !synchronized_) {
      if (aclrtSynchronizeStream(stream_) == ACL_SUCCESS) {
        synchronized_ = true;
      } else {
        return false;
      }
    }
    return true;
  }

  void mark_pending() noexcept { synchronized_ = false; }

  void release() {
    if (!synchronized_) {
      synchronize();
    }
    if (!release_synchronized_noexcept()) {
      throw AscendError(
          FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
          "Ascend development prewarm resources could not be released");
    }
  }

  [[nodiscard]] bool release_noexcept() noexcept {
    if (stream_ != nullptr && !synchronized_) {
      if (aclrtSynchronizeStream(stream_) != ACL_SUCCESS) {
        abandon_noexcept();
        return false;
      }
      synchronized_ = true;
    }
    return release_synchronized_noexcept();
  }

  [[nodiscard]] bool release_synchronized_noexcept() noexcept {
    bool success = !cleanup_failed_;
    for (DeviceAllocation& allocation : allocations_) {
      success = allocation.release_noexcept() && success;
    }
    allocations_.clear();
    success = workspace_.release_noexcept() && success;
    bindings_.clear();
    binding_shadows_.clear();
    workspace_shadow_.clear();
    reduction_input_shadow_.clear();
    if (stream_ != nullptr) {
      aclrtStream stream = stream_;
      const aclError status = aclrtDestroyStream(stream);
      stream_ = nullptr;
      success = status == ACL_SUCCESS && success;
    }
    cleanup_failed_ = !success;
    return success;
  }

  [[nodiscard]] aclrtStream stream() const noexcept { return stream_; }
  [[nodiscard]] const flagdnnBackendBindingV2* bindings() const noexcept {
    return bindings_.data();
  }
  [[nodiscard]] std::size_t binding_count() const noexcept {
    return bindings_.size();
  }
  [[nodiscard]] void* workspace() const noexcept { return workspace_.get(); }
  [[nodiscard]] std::size_t workspace_size() const noexcept {
    return workspace_.size();
  }

  void restore_stage_inputs(const AscendStageArtifact& stage) {
    require(stage.input_count == kernel_input_count(stage.kernel_family) &&
                stage.arguments.size() ==
                    kernel_runtime_argument_count(stage.kernel_family),
            "Ascend build-time stage has no input arguments",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    for (std::size_t index = 0; index < stage.input_count; ++index) {
      const DeviceRegion device = device_region(stage.arguments[index]);
      const HostRegion host = host_region(stage.arguments[index]);
      require(device.size == host.size,
              "Ascend build-time input shadow size differs",
              FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
      const std::uint8_t* source = host.pointer;
      if (stage.kernel_family == KernelFamily::kReduction) {
        require(index == 0U,
                "Ascend reduction has an unexpected input index",
                FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        reduction_input_shadow_.assign(host.size, 0xA5U);
        seed_reduction_host_input(stage, reduction_input_shadow_);
        source = reduction_input_shadow_.data();
      }
      check_acl(aclrtMemcpyAsync(device.pointer,
                                 device.size,
                                 source,
                                 host.size,
                                 ACL_MEMCPY_HOST_TO_DEVICE,
                                 stream_),
                "aclrtMemcpyAsync(restore build-time input)");
      synchronized_ = false;
    }
  }

  void reset_stage_output(const AscendStageArtifact& stage) {
    const std::size_t output_count = kernel_output_count(stage.kernel_family);
    require(stage.arguments.size() ==
                stage.input_count + output_count + 1U,
            "Ascend build-time output ABI count is invalid",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    for (std::size_t output_index = 0U; output_index < output_count;
         ++output_index) {
      const ArgumentSource& source =
          stage.arguments[stage.input_count + output_index];
      require(source.type == RawArgumentType::kPointer,
              "Ascend build-time output ABI is not a pointer",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      const DeviceRegion output =
          output_index == 0U ? stage_output(stage) : device_region(source);
      check_acl(aclrtMemsetAsync(output.pointer,
                                 output.size,
                                 0xA5,
                                 output.size,
                                 stream_),
                "aclrtMemsetAsync(autotune output sentinel)");
    }
    synchronized_ = false;
  }

  [[nodiscard]] std::vector<std::uint8_t> read_stage_output(
      const AscendStageArtifact& stage) {
    const std::size_t output_count = kernel_output_count(stage.kernel_family);
    require(stage.arguments.size() ==
                stage.input_count + output_count + 1U,
            "Ascend build-time output ABI count is invalid",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    std::vector<DeviceRegion> outputs;
    outputs.reserve(output_count);
    std::size_t snapshot_size = 0U;
    for (std::size_t output_index = 0U; output_index < output_count;
         ++output_index) {
      const ArgumentSource& source =
          stage.arguments[stage.input_count + output_index];
      require(source.type == RawArgumentType::kPointer,
              "Ascend build-time output ABI is not a pointer",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      const DeviceRegion output =
          output_index == 0U ? stage_output(stage) : device_region(source);
      require(output.size <=
                  std::numeric_limits<std::size_t>::max() - snapshot_size,
              "Ascend build-time output snapshot size overflows",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      snapshot_size += output.size;
      outputs.push_back(output);
    }
    std::vector<std::uint8_t> result(snapshot_size);
    std::size_t offset = 0U;
    for (const DeviceRegion output : outputs) {
      check_acl(aclrtMemcpyAsync(result.data() + offset,
                                 output.size,
                                 output.pointer,
                                 output.size,
                                 ACL_MEMCPY_DEVICE_TO_HOST,
                                 stream_),
                "aclrtMemcpyAsync(autotune output)");
      offset += output.size;
    }
    synchronized_ = false;
    synchronize();
    return result;
  }

  void validate_stage_inputs_unchanged(const AscendStageArtifact& stage) {
    const bool batchnorm_training =
        stage.kernel_family == KernelFamily::kBatchNorm;
    const bool batchnorm_inference =
        stage.kernel_family == KernelFamily::kBatchNormInference;
    const bool rmsnorm = stage.kernel_family == KernelFamily::kRmsNorm;
    const bool layernorm = stage.kernel_family == KernelFamily::kLayerNorm;
    const bool matmul = stage.kernel_family == KernelFamily::kMatMul;
    const bool convolution =
        stage.kernel_family == KernelFamily::kConvolutionFprop;
    if (!batchnorm_training && !batchnorm_inference && !rmsnorm &&
        !layernorm && !matmul && !convolution) {
      return;
    }
    const std::size_t expected_count =
        batchnorm_training
            ? batchnorm_kernel_input_count()
            : batchnorm_inference
                  ? batchnorm_inference_kernel_input_count()
                  : rmsnorm ? rmsnorm_kernel_input_count()
                            : layernorm ? layernorm_kernel_input_count()
                            : matmul ? matmul_kernel_input_count()
                                     : convolution_fprop_kernel_input_count();
    require(stage.input_count == expected_count,
            "Ascend structured build-time input count is invalid",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    std::vector<std::vector<std::uint8_t>> observed(expected_count);
    std::vector<HostRegion> expected(expected_count);
    for (std::size_t index = 0; index < observed.size(); ++index) {
      const DeviceRegion device = device_region(stage.arguments[index]);
      expected[index] = host_region(stage.arguments[index]);
      require(device.size == expected[index].size,
              "Ascend structured build-time input shadow size differs",
              FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
      observed[index].resize(device.size);
      check_acl(aclrtMemcpyAsync(observed[index].data(),
                                 observed[index].size(),
                                 device.pointer,
                                 device.size,
                                 ACL_MEMCPY_DEVICE_TO_HOST,
                                 stream_),
                "aclrtMemcpyAsync(validate structured input)");
      synchronized_ = false;
    }
    synchronize();
    for (std::size_t index = 0; index < observed.size(); ++index) {
      if (!std::equal(observed[index].begin(),
                      observed[index].end(),
                      expected[index].pointer)) {
        compilation_failure(
            batchnorm_training
                ? "Ascend BatchNorm training candidate modified an input during prewarm"
                : batchnorm_inference
                      ? "Ascend batchnorm candidate modified an input during prewarm"
                      : rmsnorm
                      ? "Ascend RMSNorm candidate modified an input during prewarm"
                      : layernorm
                      ? "Ascend LayerNorm candidate modified an input during prewarm"
                      : convolution
                      ? "Ascend convolution candidate modified an input during prewarm"
                      : "Ascend MatMul candidate modified an input during prewarm");
      }
    }
  }

  void validate_layout_stage_output(
      const AscendStageArtifact& stage,
      const std::vector<std::uint8_t>& actual) const {
    require(stage.kernel_family == KernelFamily::kLayout &&
                stage.input_count == 1 && stage.arguments.size() == 3 &&
                stage.tensor_storage_data_types.size() == 2 &&
                stage.input_type(0) == stage.output_type(),
            "Ascend layout build-time ABI is invalid",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    const HostRegion input = host_region(stage.arguments[0]);
    const HostRegion output = host_region(stage.arguments[1]);
    const std::size_t element_size =
        storage_element_size(stage.output_type());
    require(element_size != 0 && input.size % element_size == 0 &&
                output.size % element_size == 0 &&
                actual.size() == output.size && stage.n_elements > 0,
            "Ascend layout build-time storage is invalid",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    std::vector<bool> written(output.size / element_size, false);
    for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
      std::uint64_t input_remaining =
          static_cast<std::uint64_t>(logical);
      std::uint64_t output_remaining = input_remaining;
      std::uint64_t input_offset =
          static_cast<std::uint64_t>(stage.layout_input_base);
      std::uint64_t output_offset = 0;
      for (std::size_t reversed = stage.layout_input_dimensions.size();
           reversed != 0;
           --reversed) {
        const std::size_t axis = reversed - 1;
        const std::int64_t input_dimension =
            stage.layout_input_dimensions[axis];
        const std::int64_t output_dimension =
            stage.layout_output_dimensions[axis];
        require(input_dimension > 0 && output_dimension > 0 &&
                    stage.layout_input_strides[axis] >= 0 &&
                    stage.output_strides[axis] >= 0,
                "Ascend layout oracle metadata is invalid",
                FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        const std::uint64_t input_coordinate =
            input_remaining % static_cast<std::uint64_t>(input_dimension);
        input_remaining /= static_cast<std::uint64_t>(input_dimension);
        input_offset += input_coordinate * static_cast<std::uint64_t>(
                                               stage.layout_input_strides[axis]);
        const std::uint64_t output_coordinate =
            output_remaining % static_cast<std::uint64_t>(output_dimension);
        output_remaining /= static_cast<std::uint64_t>(output_dimension);
        output_offset += output_coordinate * static_cast<std::uint64_t>(
                                                 stage.output_strides[axis]);
      }
      require(input_remaining == 0 && output_remaining == 0 &&
                  input_offset < input.size / element_size &&
                  output_offset < output.size / element_size &&
                  !written[output_offset],
              "Ascend layout oracle tensor offset is invalid",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      const std::uint8_t* expected =
          input.pointer + input_offset * element_size;
      const std::uint8_t* observed =
          actual.data() + output_offset * element_size;
      if (std::memcmp(observed, expected, element_size) != 0) {
        compilation_failure(
            "Ascend layout candidate differs from the raw-byte host oracle");
      }
      written[output_offset] = true;
    }
    for (std::size_t element = 0; element < written.size(); ++element) {
      if (written[element]) {
        continue;
      }
      for (std::size_t byte = 0; byte < element_size; ++byte) {
        if (actual[element * element_size + byte] != 0xA5U) {
          compilation_failure(
              "Ascend layout candidate modified output padding during "
              "autotune");
        }
      }
    }
  }

  void validate_stage_output(const AscendStageArtifact& stage,
                             const std::vector<std::uint8_t>& actual) const {
    if (stage.kernel_family == KernelFamily::kLayout) {
      validate_layout_stage_output(stage, actual);
      return;
    }
    if (stage.kernel_family == KernelFamily::kReduction) {
      const HostRegion input = host_region(stage.arguments[0]);
      const HostRegion output =
          host_region(stage.arguments[reduction_output_argument_index()]);
      require(actual.size() == output.size &&
                  reduction_input_shadow_.size() == input.size,
              "Ascend reduction build-time shadow size differs",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      try {
        validate_reduction_host_output(
            stage, reduction_input_shadow_, actual);
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    if (stage.kernel_family == KernelFamily::kMatMul) {
      const HostRegion a = host_region(stage.arguments[0]);
      const HostRegion b = host_region(stage.arguments[1]);
      try {
        validate_matmul_host_output(stage,
                                    {a.pointer, a.size},
                                    {b.pointer, b.size},
                                    actual);
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    if (stage.kernel_family == KernelFamily::kConvolutionFprop) {
      const HostRegion input = host_region(stage.arguments[0]);
      const HostRegion filter = host_region(stage.arguments[1]);
      try {
        validate_convolution_fprop_host_output(
            stage,
            {input.pointer, input.size},
            {filter.pointer, filter.size},
            actual);
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    if (stage.kernel_family == KernelFamily::kBatchNorm) {
      ConstBatchNormHostBuffers inputs{};
      ConstBatchNormHostBuffers outputs{};
      for (std::size_t index = 0U; index < inputs.size(); ++index) {
        const HostRegion input = host_region(stage.arguments[index]);
        inputs[index] = std::span<const std::uint8_t>(input.pointer, input.size);
      }
      std::size_t offset = 0U;
      for (std::size_t index = 0U; index < outputs.size(); ++index) {
        const std::size_t size =
            stage.arguments[batchnorm_first_output_argument_index() + index]
                .size;
        require(size <= actual.size() - std::min(offset, actual.size()),
                "Ascend BatchNorm build-time output snapshot size differs",
                FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        outputs[index] =
            std::span<const std::uint8_t>(actual.data() + offset, size);
        offset += size;
      }
      require(offset == actual.size(),
              "Ascend BatchNorm build-time output snapshot size differs",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      try {
        validate_batchnorm_host_outputs(stage, inputs, outputs);
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    if (stage.kernel_family == KernelFamily::kBatchNormInference) {
      const HostRegion x = host_region(stage.arguments[0]);
      const HostRegion mean = host_region(stage.arguments[1]);
      const HostRegion inv_variance = host_region(stage.arguments[2]);
      const HostRegion scale = host_region(stage.arguments[3]);
      const HostRegion bias = host_region(stage.arguments[4]);
      try {
        validate_batchnorm_inference_host_output(
            stage,
            {x.pointer, x.size},
            {mean.pointer, mean.size},
            {inv_variance.pointer, inv_variance.size},
            {scale.pointer, scale.size},
            {bias.pointer, bias.size},
            actual);
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    if (stage.kernel_family == KernelFamily::kRmsNorm) {
      const HostRegion x = host_region(stage.arguments[0U]);
      const HostRegion scale = host_region(stage.arguments[1U]);
      const HostRegion bias = host_region(stage.arguments[2U]);
      const std::size_t y_size = stage.arguments[rmsnorm_y_argument_index()].size;
      const std::size_t statistic_size =
          stage.arguments[rmsnorm_inv_variance_argument_index()].size;
      require(y_size <= actual.size() &&
                  statistic_size == actual.size() - y_size,
              "Ascend RMSNorm build-time output snapshot size differs",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      try {
        validate_rmsnorm_host_outputs(
            stage,
            {x.pointer, x.size},
            {scale.pointer, scale.size},
            {bias.pointer, bias.size},
            {actual.data(), y_size},
            {actual.data() + y_size, statistic_size});
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    if (stage.kernel_family == KernelFamily::kLayerNorm) {
      const HostRegion x = host_region(stage.arguments[0U]);
      const HostRegion scale = host_region(stage.arguments[1U]);
      const HostRegion bias = host_region(stage.arguments[2U]);
      const std::size_t y_size =
          stage.arguments[layernorm_y_argument_index()].size;
      const std::size_t mean_size =
          stage.arguments[layernorm_mean_argument_index()].size;
      const std::size_t inv_size =
          stage.arguments[layernorm_inv_variance_argument_index()].size;
      require(y_size <= actual.size() &&
                  mean_size <= actual.size() - y_size &&
                  inv_size == actual.size() - y_size - mean_size,
              "Ascend LayerNorm build-time output snapshot size differs",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      try {
        validate_layernorm_host_outputs(
            stage,
            {x.pointer, x.size},
            {scale.pointer, scale.size},
            {bias.pointer, bias.size},
            {actual.data(), y_size},
            {actual.data() + y_size, mean_size},
            {actual.data() + y_size + mean_size, inv_size});
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    const HostRegion left = host_region(stage.arguments[0]);
    const bool unary = stage.kernel_family == KernelFamily::kUnary;
    const bool ternary = stage.kernel_family == KernelFamily::kTernary;
    require(unary || ternary ||
                stage.kernel_family == KernelFamily::kBinary,
            "Ascend pointwise build-time kernel family is invalid",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    const HostRegion right =
        unary ? HostRegion{} : host_region(stage.arguments[1]);
    const HostRegion predicate =
        ternary ? host_region(stage.arguments[2]) : HostRegion{};
    const HostRegion output = host_region(stage.arguments[stage.input_count]);
    require(actual.size() == output.size && stage.n_elements > 0,
            "Ascend build-time output size differs from its Graph contract",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    require(stage.tensor_storage_data_types.size() == stage.input_count + 1,
            "Ascend build-time stage tensor type count differs",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    const StorageDataType left_type = stage.input_type(0);
    const StorageDataType right_type = unary ? left_type : stage.input_type(1);
    const StorageDataType predicate_type =
        ternary ? stage.input_type(2) : StorageDataType::kBoolean;
    const StorageDataType output_type = stage.output_type();
    const std::size_t left_element_size = storage_element_size(left_type);
    const std::size_t right_element_size = storage_element_size(right_type);
    const std::size_t predicate_element_size =
        storage_element_size(predicate_type);
    const std::size_t output_element_size = storage_element_size(output_type);
    require(left.size % left_element_size == 0 &&
                (unary || right.size % right_element_size == 0) &&
                (!ternary ||
                 (predicate_type == StorageDataType::kBoolean &&
                  predicate.size % predicate_element_size == 0)) &&
                output.size % output_element_size == 0,
            "Ascend build-time tensor storage has a partial element",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    std::vector<bool> written(output.size / output_element_size, false);
    for (std::int32_t logical = 0; logical < stage.n_elements; ++logical) {
      std::uint64_t remaining = static_cast<std::uint64_t>(logical);
      std::uint64_t left_offset = 0;
      std::uint64_t right_offset = 0;
      std::uint64_t predicate_offset = 0;
      std::uint64_t output_offset = 0;
      for (std::size_t reversed = stage.dimensions.size(); reversed != 0;
           --reversed) {
        const std::size_t axis = reversed - 1;
        const std::int64_t dimension = stage.dimensions[axis];
        require(dimension > 0,
                "Ascend build-time oracle has a nonpositive dimension",
                FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        const std::uint64_t coordinate =
            remaining % static_cast<std::uint64_t>(dimension);
        remaining /= static_cast<std::uint64_t>(dimension);
        left_offset += coordinate *
                       static_cast<std::uint64_t>(stage.left_strides[axis]);
        if (!unary) {
          right_offset += coordinate * static_cast<std::uint64_t>(
                                         stage.right_strides[axis]);
        }
        if (ternary) {
          predicate_offset += coordinate * static_cast<std::uint64_t>(
                                             stage.mask_strides[axis]);
        }
        output_offset += coordinate *
                         static_cast<std::uint64_t>(stage.output_strides[axis]);
      }
      require(remaining == 0 &&
                  left_offset < left.size / left_element_size &&
                  (unary || right_offset < right.size / right_element_size) &&
                  (!ternary ||
                   predicate_offset <
                       predicate.size / predicate_element_size) &&
                  output_offset < output.size / output_element_size,
              "Ascend build-time oracle tensor offset is out of range",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      const float left_value = decode_storage_value(
          left.pointer + left_offset * left_element_size, left_type);
      const float right_value =
          unary ? 0.0F
                : decode_storage_value(right.pointer +
                                           right_offset * right_element_size,
                                       right_type);
      const float predicate_value =
          ternary ? decode_storage_value(
                        predicate.pointer +
                            predicate_offset * predicate_element_size,
                        predicate_type)
                  : 0.0F;
      const std::uint8_t* selected_storage = nullptr;
      float expected = 0.0F;
      if (stage.operation == "sqrt") {
        expected = std::sqrt(left_value);
      } else if (stage.operation == "erf") {
        expected = std::erf(left_value);
      } else if (stage.operation == "rsqrt") {
        expected = 1.0F / std::sqrt(left_value);
      } else if (stage.operation == "reciprocal") {
        expected = 1.0F / left_value;
      } else if (stage.operation == "exp") {
        expected = std::exp(left_value);
      } else if (stage.operation == "log") {
        expected = std::log(left_value);
      } else if (stage.operation == "cos") {
        expected = std::cos(left_value);
      } else if (stage.operation == "sin") {
        expected = std::sin(left_value);
      } else if (stage.operation == "tan") {
        expected = std::tan(left_value);
      } else if (stage.operation == "sigmoid") {
        expected = 1.0F / (1.0F + std::exp(-left_value));
      } else if (stage.operation == "tanh") {
        expected =
            2.0F / (1.0F + std::exp(-2.0F * left_value)) - 1.0F;
      } else if (stage.operation == "elu") {
        expected =
            left_value > 0.0F
                ? left_value
                : static_cast<float>(stage.elu_alpha) *
                      (std::exp(left_value) - 1.0F);
      } else if (stage.operation == "gelu") {
        expected =
            0.5F * left_value *
            (1.0F + std::erf(left_value * 0.7071067811865476F));
      } else if (stage.operation == "softplus") {
        const float beta = static_cast<float>(stage.softplus_beta);
        const float scaled = beta * left_value;
        expected =
            (std::fmax(scaled, 0.0F) +
             std::log1p(std::exp(-std::fabs(scaled)))) /
            beta;
      } else if (stage.operation == "swish") {
        const float scaled =
            static_cast<float>(stage.swish_beta) * left_value;
        expected = left_value / (1.0F + std::exp(-scaled));
      } else if (stage.operation == "gelu_approx_tanh") {
        const float cubic = left_value * left_value * left_value;
        const float approximate_argument =
            0.7978845608028654F * (left_value + 0.044715F * cubic);
        const float approximate_tanh =
            2.0F / (1.0F + std::exp(-2.0F * approximate_argument)) - 1.0F;
        expected = 0.5F * left_value * (1.0F + approximate_tanh);
      } else if (stage.operation == "identity") {
        expected = left_value;
      } else if (stage.operation == "neg") {
        expected = -left_value;
      } else if (stage.operation == "abs") {
        expected = std::fabs(left_value);
      } else if (stage.operation == "ceil") {
        expected = std::ceil(left_value);
      } else if (stage.operation == "floor") {
        expected = std::floor(left_value);
      } else if (stage.operation == "relu") {
        expected =
            left_value < static_cast<float>(stage.lower_clip)
                ? static_cast<float>(stage.lower_clip) +
                      static_cast<float>(stage.negative_slope) *
                          (left_value - static_cast<float>(stage.lower_clip))
                : left_value;
        if (stage.has_upper_clip) {
          expected = std::fmin(expected,
                               static_cast<float>(stage.upper_clip));
        }
      } else if (stage.operation == "add") {
        expected =
            left_value + static_cast<float>(stage.alpha) * right_value;
      } else if (stage.operation == "sub") {
        expected =
            left_value - static_cast<float>(stage.alpha) * right_value;
      } else if (stage.operation == "mul") {
        expected = left_value * right_value;
      } else if (stage.operation == "div") {
        expected = left_value / right_value;
      } else if (stage.operation == "min") {
        expected = std::fmin(left_value, right_value);
      } else if (stage.operation == "max") {
        expected = std::fmax(left_value, right_value);
      } else if (stage.operation == "mod") {
        expected = std::fmod(left_value, right_value);
      } else if (stage.operation == "pow") {
        expected = std::pow(left_value, right_value);
      } else if (stage.operation == "cmp_eq") {
        expected = left_value == right_value ? 1.0F : 0.0F;
      } else if (stage.operation == "cmp_neq") {
        expected = left_value != right_value ? 1.0F : 0.0F;
      } else if (stage.operation == "cmp_gt") {
        expected = left_value > right_value ? 1.0F : 0.0F;
      } else if (stage.operation == "cmp_ge") {
        expected = left_value >= right_value ? 1.0F : 0.0F;
      } else if (stage.operation == "cmp_lt") {
        expected = left_value < right_value ? 1.0F : 0.0F;
      } else if (stage.operation == "cmp_le") {
        expected = left_value <= right_value ? 1.0F : 0.0F;
      } else if (stage.operation == "sigmoid_backward") {
        const float e = std::exp(-std::fabs(right_value));
        const float inv = 1.0F / (1.0F + e);
        expected = left_value * e * inv * inv;
      } else if (stage.operation == "logical_not") {
        expected = left_value == 0.0F ? 1.0F : 0.0F;
      } else if (stage.operation == "logical_and") {
        expected = left_value != 0.0F && right_value != 0.0F ? 1.0F : 0.0F;
      } else if (stage.operation == "logical_or") {
        expected = left_value != 0.0F || right_value != 0.0F ? 1.0F : 0.0F;
      } else if (stage.operation == "binary_select") {
        selected_storage =
            predicate_value != 0.0F
                ? left.pointer + left_offset * left_element_size
                : right.pointer + right_offset * right_element_size;
        expected = decode_storage_value(selected_storage, output_type);
      } else {
        compilation_failure(
            "Ascend build-time pointwise oracle received an unsupported "
            "operation");
      }
      const std::uint8_t* observed_storage =
          actual.data() + output_offset * output_element_size;
      if (output_type == StorageDataType::kBoolean) {
        const std::uint8_t observed_byte = observed_storage[0];
        if (observed_byte != 0U && observed_byte != 1U) {
          compilation_failure(
              "Ascend BOOLEAN candidate output is not canonical 0/1");
        }
      }
      const float observed =
          decode_storage_value(observed_storage, output_type);
      const bool transcendental = stage.operation == "exp" ||
                                  stage.operation == "log" ||
                                  stage.operation == "erf" ||
                                  stage.operation == "cos" ||
                                  stage.operation == "sin" ||
                                  stage.operation == "tan" ||
                                  stage.operation == "sigmoid" ||
                                  stage.operation == "tanh" ||
                                  stage.operation == "elu" ||
                                  stage.operation == "gelu" ||
                                  stage.operation == "softplus" ||
                                  stage.operation == "swish" ||
                                  stage.operation == "gelu_approx_tanh";
      const bool binary_transcendental =
          stage.operation == "mod" || stage.operation == "pow" ||
          stage.operation == "sigmoid_backward";
      const float absolute_tolerance =
          transcendental || binary_transcendental
              ? output_type == StorageDataType::kBFloat16
                    ? 5.0e-2F
                    : 2.0e-2F
              : output_type == StorageDataType::kFloat32
                    ? 1.0e-5F
                    : output_type == StorageDataType::kFloat16
                          ? 5.0e-3F
                          : 5.0e-2F;
      const float relative_tolerance =
          transcendental || binary_transcendental ? 1.0e-2F
                                                   : absolute_tolerance;
      const bool matches =
          stage.operation == "binary_select"
              ? selected_storage != nullptr &&
                    left_element_size == output_element_size &&
                    right_element_size == output_element_size &&
                    std::memcmp(observed_storage,
                                selected_storage,
                                output_element_size) == 0
          : output_type == StorageDataType::kBoolean
              ? observed == expected
              : std::isnan(expected)
              ? std::isnan(observed)
              : std::isfinite(expected)
                    ? std::isfinite(observed) &&
                          std::abs(observed - expected) <=
                              absolute_tolerance +
                                  relative_tolerance * std::abs(expected)
                    : observed == expected;
      if (!matches) {
        compilation_failure(
            "Ascend candidate differs from the build-time host pointwise "
            "oracle");
      }
      written[output_offset] = true;
    }
    for (std::size_t element = 0; element < written.size(); ++element) {
      if (written[element]) {
        continue;
      }
      for (std::size_t byte = 0; byte < output_element_size; ++byte) {
        if (actual[element * output_element_size + byte] != 0xA5U) {
          compilation_failure(
              "Ascend candidate modified output padding during autotune");
        }
      }
    }
  }

  void commit_stage_output(const AscendStageArtifact& stage,
                           const std::vector<std::uint8_t>& actual) {
    if (stage.kernel_family == KernelFamily::kBatchNorm) {
      ConstBatchNormHostBuffers source{};
      BatchNormHostBuffers destination{};
      std::size_t offset = 0U;
      for (std::size_t index = 0U; index < source.size(); ++index) {
        MutableHostRegion output = mutable_host_region(
            stage.arguments[batchnorm_first_output_argument_index() + index]);
        require(output.size <= actual.size() - std::min(offset, actual.size()),
                "Ascend BatchNorm build-time output commit size differs",
                FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
        source[index] =
            std::span<const std::uint8_t>(actual.data() + offset, output.size);
        destination[index] =
            std::span<std::uint8_t>(output.pointer, output.size);
        offset += output.size;
      }
      require(offset == actual.size(),
              "Ascend BatchNorm build-time output commit size differs",
              FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
      try {
        commit_batchnorm_host_outputs(stage, source, destination);
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    if (stage.kernel_family == KernelFamily::kLayerNorm) {
      MutableHostRegion y =
          mutable_host_region(stage.arguments[layernorm_y_argument_index()]);
      MutableHostRegion mean =
          mutable_host_region(stage.arguments[layernorm_mean_argument_index()]);
      MutableHostRegion inv_variance = mutable_host_region(
          stage.arguments[layernorm_inv_variance_argument_index()]);
      require(y.size <= actual.size() &&
                  mean.size <= actual.size() - y.size &&
                  inv_variance.size == actual.size() - y.size - mean.size,
              "Ascend LayerNorm build-time output commit size differs",
              FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
      try {
        commit_layernorm_host_outputs(
            stage,
            {actual.data(), y.size},
            {actual.data() + y.size, mean.size},
            {actual.data() + y.size + mean.size, inv_variance.size},
            {y.pointer, y.size},
            {mean.pointer, mean.size},
            {inv_variance.pointer, inv_variance.size});
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    if (stage.kernel_family == KernelFamily::kRmsNorm) {
      MutableHostRegion y =
          mutable_host_region(stage.arguments[rmsnorm_y_argument_index()]);
      MutableHostRegion inv_variance = mutable_host_region(
          stage.arguments[rmsnorm_inv_variance_argument_index()]);
      require(y.size <= actual.size() &&
                  inv_variance.size == actual.size() - y.size,
              "Ascend RMSNorm build-time output commit size differs",
              FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
      try {
        commit_rmsnorm_host_outputs(
            stage,
            {actual.data(), y.size},
            {actual.data() + y.size, inv_variance.size},
            {y.pointer, y.size},
            {inv_variance.pointer, inv_variance.size});
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    MutableHostRegion output =
        mutable_host_region(stage.arguments[stage.input_count]);
    require(actual.size() == output.size,
            "Ascend build-time output commit size differs",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
    if (stage.kernel_family == KernelFamily::kReduction) {
      try {
        commit_reduction_host_output(
            stage, actual, {output.pointer, output.size});
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    if (stage.kernel_family == KernelFamily::kMatMul) {
      try {
        commit_matmul_host_output(
            stage, actual, {output.pointer, output.size});
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    if (stage.kernel_family == KernelFamily::kConvolutionFprop) {
      try {
        commit_convolution_fprop_host_output(
            stage, actual, {output.pointer, output.size});
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    if (stage.kernel_family == KernelFamily::kBatchNormInference) {
      try {
        commit_batchnorm_inference_host_output(
            stage, actual, {output.pointer, output.size});
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      return;
    }
    std::copy(actual.begin(), actual.end(), output.pointer);
  }

 private:
  struct DeviceRegion {
    void* pointer = nullptr;
    std::size_t size = 0;
  };

  struct HostRegion {
    const std::uint8_t* pointer = nullptr;
    std::size_t size = 0;
  };

  struct MutableHostRegion {
    std::uint8_t* pointer = nullptr;
    std::size_t size = 0;
  };

  [[nodiscard]] std::size_t binding_index(std::int64_t uid) const {
    for (std::size_t index = 0; index < bindings_.size(); ++index) {
      if (bindings_[index].uid == uid) {
        return index;
      }
    }
    compilation_failure("Ascend build-time binding shadow is missing");
  }

  [[nodiscard]] HostRegion host_region(
      const ArgumentSource& source) const {
    if (source.source == ArgumentSourceKind::kBinding) {
      const std::size_t index = binding_index(source.uid);
      require(index < binding_shadows_.size() &&
                  source.size <= binding_shadows_[index].size(),
              "Ascend build-time binding shadow range is invalid",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      return {binding_shadows_[index].data(), source.size};
    }
    if (source.source == ArgumentSourceKind::kGraphWorkspace) {
      require(source.workspace_offset <= workspace_shadow_.size() &&
                  source.size <=
                      workspace_shadow_.size() - source.workspace_offset,
              "Ascend build-time workspace shadow range is invalid",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      return {workspace_shadow_.data() + source.workspace_offset,
              source.size};
    }
    compilation_failure("Ascend build-time pointer shadow cannot be a scalar");
  }

  [[nodiscard]] MutableHostRegion mutable_host_region(
      const ArgumentSource& source) {
    if (source.source == ArgumentSourceKind::kBinding) {
      const std::size_t index = binding_index(source.uid);
      require(index < binding_shadows_.size() &&
                  source.size <= binding_shadows_[index].size(),
              "Ascend build-time binding shadow range is invalid",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      return {binding_shadows_[index].data(), source.size};
    }
    if (source.source == ArgumentSourceKind::kGraphWorkspace) {
      require(source.workspace_offset <= workspace_shadow_.size() &&
                  source.size <=
                      workspace_shadow_.size() - source.workspace_offset,
              "Ascend build-time workspace shadow range is invalid",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      return {workspace_shadow_.data() + source.workspace_offset,
              source.size};
    }
    compilation_failure("Ascend build-time pointer shadow cannot be a scalar");
  }

  void seed_external_inputs(const AscendArtifact& artifact) {
    std::set<std::int64_t> written_bindings;
    for (const AscendStageArtifact& stage : artifact.stages) {
      const std::size_t output_count = kernel_output_count(stage.kernel_family);
      for (std::size_t output = 0U; output < output_count; ++output) {
        const std::size_t argument_index = stage.input_count + output;
        if (stage.arguments.size() > argument_index &&
            stage.arguments[argument_index].source ==
                ArgumentSourceKind::kBinding) {
          written_bindings.insert(stage.arguments[argument_index].uid);
        }
      }
    }
    std::set<std::int64_t> positive_input_bindings;
    std::set<std::int64_t> seeded;
    for (const AscendStageArtifact& stage : artifact.stages) {
      if (stage.kernel_family == KernelFamily::kConvolutionFprop) {
        std::array<std::vector<std::uint8_t>, 2> seeded_inputs;
        seeded_inputs[0].assign(stage.arguments[0].size, 0xA5U);
        seeded_inputs[1].assign(stage.arguments[1].size, 0xA5U);
        try {
          seed_convolution_fprop_host_inputs(
              stage, seeded_inputs[0], seeded_inputs[1]);
        } catch (const std::invalid_argument& error) {
          compilation_failure(error.what());
        }
        for (std::size_t index = 0U; index < seeded_inputs.size(); ++index) {
          const ArgumentSource& argument = stage.arguments[index];
          if (argument.source != ArgumentSourceKind::kBinding ||
              written_bindings.contains(argument.uid) ||
              !seeded.insert(argument.uid).second) {
            continue;
          }
          const std::size_t binding = binding_index(argument.uid);
          require(binding_shadows_[binding].size() == seeded_inputs[index].size(),
                  "Ascend convolution seed shadow size differs",
                  FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
          binding_shadows_[binding] = seeded_inputs[index];
          check_acl(aclrtMemcpy(allocations_[binding].get(),
                                allocations_[binding].size(),
                                binding_shadows_[binding].data(),
                                binding_shadows_[binding].size(),
                                ACL_MEMCPY_HOST_TO_DEVICE),
                    "aclrtMemcpy(build-time convolution input)");
        }
        continue;
      }
      if (stage.kernel_family == KernelFamily::kMatMul) {
        std::array<std::vector<std::uint8_t>, 2> seeded_inputs;
        seeded_inputs[0].assign(stage.arguments[0].size, 0xA5U);
        seeded_inputs[1].assign(stage.arguments[1].size, 0xA5U);
        try {
          seed_matmul_host_inputs(
              stage, seeded_inputs[0], seeded_inputs[1]);
        } catch (const std::invalid_argument& error) {
          compilation_failure(error.what());
        }
        for (std::size_t index = 0U; index < seeded_inputs.size(); ++index) {
          const ArgumentSource& argument = stage.arguments[index];
          if (argument.source != ArgumentSourceKind::kBinding ||
              written_bindings.contains(argument.uid) ||
              !seeded.insert(argument.uid).second) {
            continue;
          }
          const std::size_t binding = binding_index(argument.uid);
          require(binding_shadows_[binding].size() == seeded_inputs[index].size(),
                  "Ascend MatMul seed shadow size differs",
                  FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
          binding_shadows_[binding] = seeded_inputs[index];
          check_acl(aclrtMemcpy(allocations_[binding].get(),
                                allocations_[binding].size(),
                                binding_shadows_[binding].data(),
                                binding_shadows_[binding].size(),
                                ACL_MEMCPY_HOST_TO_DEVICE),
                    "aclrtMemcpy(build-time MatMul input)");
        }
        continue;
      }
      if (stage.kernel_family == KernelFamily::kBatchNorm) {
        std::array<std::vector<std::uint8_t>, 5> seeded_inputs;
        BatchNormHostBuffers seeded_spans{};
        for (std::size_t index = 0U; index < seeded_inputs.size(); ++index) {
          seeded_inputs[index].assign(stage.arguments[index].size, 0xA5U);
          seeded_spans[index] = std::span<std::uint8_t>(seeded_inputs[index]);
        }
        try {
          seed_batchnorm_host_inputs(stage, seeded_spans);
        } catch (const std::invalid_argument& error) {
          compilation_failure(error.what());
        }
        for (std::size_t index = 0U; index < seeded_inputs.size(); ++index) {
          const ArgumentSource& argument = stage.arguments[index];
          if (argument.source != ArgumentSourceKind::kBinding ||
              written_bindings.contains(argument.uid) ||
              !seeded.insert(argument.uid).second) {
            continue;
          }
          const std::size_t binding = binding_index(argument.uid);
          require(binding_shadows_[binding].size() == seeded_inputs[index].size(),
                  "Ascend BatchNorm seed shadow size differs",
                  FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
          binding_shadows_[binding] = seeded_inputs[index];
          check_acl(aclrtMemcpy(allocations_[binding].get(),
                                allocations_[binding].size(),
                                binding_shadows_[binding].data(),
                                binding_shadows_[binding].size(),
                                ACL_MEMCPY_HOST_TO_DEVICE),
                    "aclrtMemcpy(build-time BatchNorm input)");
        }
        continue;
      }
      if (stage.kernel_family == KernelFamily::kLayerNorm) {
        std::array<std::vector<std::uint8_t>, 3> seeded_inputs;
        for (std::size_t index = 0U; index < seeded_inputs.size(); ++index) {
          seeded_inputs[index].assign(stage.arguments[index].size, 0xA5U);
        }
        try {
          seed_layernorm_host_inputs(stage,
                                     seeded_inputs[0U],
                                     seeded_inputs[1U],
                                     seeded_inputs[2U]);
        } catch (const std::invalid_argument& error) {
          compilation_failure(error.what());
        }
        for (std::size_t index = 0U; index < seeded_inputs.size(); ++index) {
          const ArgumentSource& argument = stage.arguments[index];
          if (argument.source != ArgumentSourceKind::kBinding ||
              written_bindings.contains(argument.uid) ||
              !seeded.insert(argument.uid).second) {
            continue;
          }
          const std::size_t binding = binding_index(argument.uid);
          require(binding_shadows_[binding].size() == seeded_inputs[index].size(),
                  "Ascend LayerNorm seed shadow size differs",
                  FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
          binding_shadows_[binding] = seeded_inputs[index];
          check_acl(aclrtMemcpy(allocations_[binding].get(),
                                allocations_[binding].size(),
                                binding_shadows_[binding].data(),
                                binding_shadows_[binding].size(),
                                ACL_MEMCPY_HOST_TO_DEVICE),
                    "aclrtMemcpy(build-time LayerNorm input)");
        }
        continue;
      }
      if (stage.kernel_family == KernelFamily::kRmsNorm) {
        std::array<std::vector<std::uint8_t>, 3> seeded_inputs;
        for (std::size_t index = 0U; index < seeded_inputs.size(); ++index) {
          seeded_inputs[index].assign(stage.arguments[index].size, 0xA5U);
        }
        try {
          seed_rmsnorm_host_inputs(stage,
                                   seeded_inputs[0U],
                                   seeded_inputs[1U],
                                   seeded_inputs[2U]);
        } catch (const std::invalid_argument& error) {
          compilation_failure(error.what());
        }
        for (std::size_t index = 0U; index < seeded_inputs.size(); ++index) {
          const ArgumentSource& argument = stage.arguments[index];
          if (argument.source != ArgumentSourceKind::kBinding ||
              written_bindings.contains(argument.uid) ||
              !seeded.insert(argument.uid).second) {
            continue;
          }
          const std::size_t binding = binding_index(argument.uid);
          require(binding_shadows_[binding].size() == seeded_inputs[index].size(),
                  "Ascend RMSNorm seed shadow size differs",
                  FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
          binding_shadows_[binding] = seeded_inputs[index];
          check_acl(aclrtMemcpy(allocations_[binding].get(),
                                allocations_[binding].size(),
                                binding_shadows_[binding].data(),
                                binding_shadows_[binding].size(),
                                ACL_MEMCPY_HOST_TO_DEVICE),
                    "aclrtMemcpy(build-time RMSNorm input)");
        }
        continue;
      }
      if (stage.kernel_family != KernelFamily::kBatchNormInference) {
        continue;
      }
      std::array<std::vector<std::uint8_t>, 5> seeded_inputs;
      for (std::size_t index = 0; index < seeded_inputs.size(); ++index) {
        seeded_inputs[index].assign(stage.arguments[index].size, 0xA5U);
      }
      try {
        seed_batchnorm_inference_host_inputs(stage,
                                             seeded_inputs[0],
                                             seeded_inputs[1],
                                             seeded_inputs[2],
                                             seeded_inputs[3],
                                             seeded_inputs[4]);
      } catch (const std::invalid_argument& error) {
        compilation_failure(error.what());
      }
      for (std::size_t index = 0; index < seeded_inputs.size(); ++index) {
        const ArgumentSource& argument = stage.arguments[index];
        if (index != 0U) {
          require(argument.source == ArgumentSourceKind::kBinding &&
                      !written_bindings.contains(argument.uid),
                  "Ascend batchnorm parameter must be an external binding",
                  FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        }
        if (argument.source != ArgumentSourceKind::kBinding ||
            written_bindings.contains(argument.uid) ||
            !seeded.insert(argument.uid).second) {
          continue;
        }
        const std::size_t binding = binding_index(argument.uid);
        require(binding_shadows_[binding].size() == seeded_inputs[index].size(),
                "Ascend batchnorm seed shadow size differs",
                FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        binding_shadows_[binding] = seeded_inputs[index];
        check_acl(aclrtMemcpy(allocations_[binding].get(),
                              allocations_[binding].size(),
                              binding_shadows_[binding].data(),
                              binding_shadows_[binding].size(),
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(build-time batchnorm input)");
      }
    }
    const bool has_workspace_power_base =
        std::any_of(artifact.stages.begin(),
                    artifact.stages.end(),
                    [](const AscendStageArtifact& stage) {
                      return stage.operation == "pow" &&
                             stage.arguments[0].source ==
                                 ArgumentSourceKind::kGraphWorkspace;
                    });
    for (const AscendStageArtifact& stage : artifact.stages) {
      if (has_workspace_power_base) {
        for (std::size_t argument_index = 0;
             argument_index < stage.input_count;
             ++argument_index) {
          const ArgumentSource& argument = stage.arguments[argument_index];
          if (argument.source == ArgumentSourceKind::kBinding) {
            positive_input_bindings.insert(argument.uid);
          }
        }
      }
      if (stage.operation == "pow") {
        const ArgumentSource& base = stage.arguments[0];
        if (base.source == ArgumentSourceKind::kBinding) {
          positive_input_bindings.insert(base.uid);
        }
        continue;
      }
      if (stage.operation != "sqrt" && stage.operation != "rsqrt" &&
          stage.operation != "reciprocal" && stage.operation != "log") {
        continue;
      }
      for (std::size_t argument_index = 0;
           argument_index < stage.input_count;
           ++argument_index) {
        const ArgumentSource& argument = stage.arguments[argument_index];
        if (argument.source == ArgumentSourceKind::kBinding) {
          positive_input_bindings.insert(argument.uid);
        }
      }
    }
    for (const AscendStageArtifact& stage : artifact.stages) {
      for (std::size_t argument_index = 0;
           argument_index < stage.input_count;
           ++argument_index) {
        const ArgumentSource& argument = stage.arguments[argument_index];
        if (argument.source != ArgumentSourceKind::kBinding ||
            written_bindings.contains(argument.uid) ||
            !seeded.insert(argument.uid).second) {
          continue;
        }
        const std::size_t index = binding_index(argument.uid);
        fill_storage_pattern(binding_shadows_[index],
                             stage.input_type(argument_index),
                             argument.uid,
                             positive_input_bindings.contains(argument.uid),
                             stage.operation == "sigmoid_backward" &&
                                 argument_index == 1,
                             stage.operation.starts_with("cmp_")
                                 ? static_cast<int>(argument_index)
                                 : -1);
        check_acl(aclrtMemcpy(allocations_[index].get(),
                              allocations_[index].size(),
                              binding_shadows_[index].data(),
                              binding_shadows_[index].size(),
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(build-time nonzero input)");
      }
    }
  }

  [[nodiscard]] DeviceRegion stage_output(
      const AscendStageArtifact& stage) const {
    const std::string_view expected_name =
        stage.kernel_family == KernelFamily::kLayout
            ? "output_ptr"
            : stage.kernel_family == KernelFamily::kReduction
                  ? reduction_output_argument_name()
            : stage.kernel_family == KernelFamily::kMatMul
                  ? matmul_output_argument_name()
            : stage.kernel_family == KernelFamily::kConvolutionFprop
                  ? convolution_fprop_output_argument_name()
                  : stage.kernel_family == KernelFamily::kBatchNorm
                        ? "y_ptr"
                  : stage.kernel_family == KernelFamily::kBatchNormInference
                        ? batchnorm_inference_output_argument_name()
                        : stage.kernel_family == KernelFamily::kRmsNorm
                              ? "y_ptr"
                        : stage.kernel_family == KernelFamily::kLayerNorm
                              ? "y_ptr"
                        : "out_ptr";
    require(stage.input_count == kernel_input_count(stage.kernel_family) &&
                stage.arguments.size() > stage.input_count &&
                stage.arguments[stage.input_count].name == expected_name &&
                stage.arguments[stage.input_count].type ==
                    RawArgumentType::kPointer,
            "Ascend autotune stage has no exact output argument",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    return device_region(stage.arguments[stage.input_count]);
  }

  [[nodiscard]] DeviceRegion rmsnorm_statistic_output(
      const AscendStageArtifact& stage) const {
    require(stage.kernel_family == KernelFamily::kRmsNorm &&
                stage.input_count == rmsnorm_kernel_input_count() &&
                stage.arguments.size() == rmsnorm_runtime_argument_count() &&
                stage.arguments[rmsnorm_y_argument_index()].name == "y_ptr" &&
                stage.arguments[rmsnorm_inv_variance_argument_index()].name ==
                    "inv_variance_ptr" &&
                stage.arguments[rmsnorm_inv_variance_argument_index()].type ==
                    RawArgumentType::kPointer,
            "Ascend RMSNorm stage has no exact statistic output argument",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    return device_region(
        stage.arguments[rmsnorm_inv_variance_argument_index()]);
  }

  [[nodiscard]] DeviceRegion layernorm_mean_output(
      const AscendStageArtifact& stage) const {
    require(stage.kernel_family == KernelFamily::kLayerNorm &&
                stage.input_count == layernorm_kernel_input_count() &&
                stage.arguments.size() == layernorm_runtime_argument_count() &&
                stage.arguments[layernorm_y_argument_index()].name == "y_ptr" &&
                stage.arguments[layernorm_mean_argument_index()].name ==
                    "mean_ptr" &&
                stage.arguments[layernorm_mean_argument_index()].type ==
                    RawArgumentType::kPointer,
            "Ascend LayerNorm stage has no exact mean output argument",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    return device_region(stage.arguments[layernorm_mean_argument_index()]);
  }

  [[nodiscard]] DeviceRegion layernorm_inv_variance_output(
      const AscendStageArtifact& stage) const {
    require(stage.kernel_family == KernelFamily::kLayerNorm &&
                stage.input_count == layernorm_kernel_input_count() &&
                stage.arguments.size() == layernorm_runtime_argument_count() &&
                stage.arguments[layernorm_inv_variance_argument_index()].name ==
                    "inv_variance_ptr" &&
                stage.arguments[layernorm_inv_variance_argument_index()].type ==
                    RawArgumentType::kPointer,
            "Ascend LayerNorm stage has no exact inverse variance output argument",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    return device_region(
        stage.arguments[layernorm_inv_variance_argument_index()]);
  }

  [[nodiscard]] DeviceRegion device_region(
      const ArgumentSource& source) const {
    if (source.source == ArgumentSourceKind::kBinding) {
      for (std::size_t index = 0; index < bindings_.size(); ++index) {
        if (bindings_[index].uid == source.uid) {
          require(index < allocations_.size() &&
                      source.size <= allocations_[index].size(),
                  "Ascend autotune output binding exceeds its allocation",
                  FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
          return {bindings_[index].device_pointer, source.size};
        }
      }
      compilation_failure("Ascend autotune output binding is missing");
    }
    if (source.source == ArgumentSourceKind::kGraphWorkspace) {
      require(workspace_.get() != nullptr &&
                  source.workspace_offset <= workspace_.size() &&
                  source.size <= workspace_.size() - source.workspace_offset,
              "Ascend autotune output workspace range is invalid",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      return {static_cast<std::uint8_t*>(workspace_.get()) +
                  source.workspace_offset,
              source.size};
    }
    compilation_failure("Ascend build-time pointer region cannot be a scalar");
  }

  void abandon_noexcept() noexcept {
    for (DeviceAllocation& allocation : allocations_) {
      allocation.abandon();
    }
    allocations_.clear();
    workspace_.abandon();
    bindings_.clear();
    binding_shadows_.clear();
    workspace_shadow_.clear();
    reduction_input_shadow_.clear();
    stream_ = nullptr;
    synchronized_ = false;
    cleanup_failed_ = true;
  }

  aclrtStream stream_ = nullptr;
  std::vector<DeviceAllocation> allocations_;
  std::vector<flagdnnBackendBindingV2> bindings_;
  std::vector<std::vector<std::uint8_t>> binding_shadows_;
  DeviceAllocation workspace_;
  std::vector<std::uint8_t> workspace_shadow_;
  std::vector<std::uint8_t> reduction_input_shadow_;
  bool synchronized_ = false;
  bool cleanup_failed_ = false;
};

struct HookCapture {
  bool entered = false;
  unsigned int shared_memory = 0;
};

void require_locked_configuration(const EngineBuildContext& context,
                                  bool* terminal_failure) {
  if (detail::process_domain().terminal_failure_latched() ||
      !process_configuration_matches(context)) {
    *terminal_failure = true;
    latch_process_terminal();
    throw AscendError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "Ascend process configuration changed while waiting for or "
        "executing a raw launch");
  }
}

void clear_launch_hooks_noexcept() noexcept {
  try {
    triton_jit::clear_launch_hooks();
  } catch (...) {
    latch_process_terminal();
  }
}

class LaunchHookOwner {
 public:
  LaunchHookOwner(const EngineBuildContext& context,
                  const LtjNpuRawCandidate& candidate,
                  aclrtStream stream,
                  HookCapture* capture) {
    triton_jit::set_launch_enter_hook(
        [&context, &candidate, stream, capture](
            const triton_jit::LaunchMetadata& metadata) {
          aclrtContext current = nullptr;
          aclError status = aclrtGetCurrentContext(&current);
          if (status == ACL_SUCCESS && current == nullptr) {
            status = aclrtSetCurrentContext(context.context);
            current = status == ACL_SUCCESS ? context.context : nullptr;
          }
          const bool matches =
              !capture->entered &&
              !detail::process_domain().terminal_failure_latched() &&
              process_configuration_matches(context) &&
              status == ACL_SUCCESS &&
              current == context.context &&
              metadata.kernel_name == candidate.entry_point &&
              metadata.grid_x == candidate.grid[0] &&
              metadata.grid_y == candidate.grid[1] &&
              metadata.grid_z == candidate.grid[2] &&
              metadata.num_warps == static_cast<int>(candidate.num_warps) &&
              metadata.signature == candidate.full_signature &&
              metadata.stream == reinterpret_cast<void*>(stream);
          if (!matches) {
            latch_process_terminal();
            throw std::runtime_error(
                "libtriton_jit launch-enter metadata/context mismatch");
          }
          capture->entered = true;
          capture->shared_memory = metadata.shared_memory;
        });
    active_ = true;
  }

  ~LaunchHookOwner() { clear_noexcept(); }
  LaunchHookOwner(const LaunchHookOwner&) = delete;
  LaunchHookOwner& operator=(const LaunchHookOwner&) = delete;

  void clear() {
    if (active_) {
      triton_jit::clear_launch_hooks();
      active_ = false;
    }
  }

  void clear_noexcept() noexcept {
    if (active_) {
      clear_launch_hooks_noexcept();
      active_ = false;
    }
  }

 private:
  bool active_ = false;
};

struct StageLaunch {
  std::size_t stage_index = 0;
  LtjNpuRawCandidate candidate;
  LtjFunction* function = nullptr;
};

class AutotuneEvents {
 public:
  AutotuneEvents() {
    check_acl(aclrtCreateEventExWithFlag(&start_, ACL_EVENT_TIME_LINE),
              "aclrtCreateEventExWithFlag(autotune start)");
    try {
      check_acl(aclrtCreateEventExWithFlag(&end_, ACL_EVENT_TIME_LINE),
                "aclrtCreateEventExWithFlag(autotune end)");
    } catch (...) {
      (void)aclrtDestroyEvent(start_);
      start_ = nullptr;
      throw;
    }
  }

  ~AutotuneEvents() {
    if (!release_noexcept()) {
      latch_process_terminal();
    }
  }

  AutotuneEvents(const AutotuneEvents&) = delete;
  AutotuneEvents& operator=(const AutotuneEvents&) = delete;

  [[nodiscard]] aclrtEvent start() const noexcept { return start_; }
  [[nodiscard]] aclrtEvent end() const noexcept { return end_; }

  void abandon() noexcept {
    start_ = nullptr;
    end_ = nullptr;
  }

  void release() {
    bool success = true;
    if (end_ != nullptr) {
      const aclrtEvent event = end_;
      end_ = nullptr;
      success = aclrtDestroyEvent(event) == ACL_SUCCESS && success;
    }
    if (start_ != nullptr) {
      const aclrtEvent event = start_;
      start_ = nullptr;
      success = aclrtDestroyEvent(event) == ACL_SUCCESS && success;
    }
    if (!success) {
      throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                        "Ascend autotune events could not be destroyed");
    }
  }

 private:
  [[nodiscard]] bool release_noexcept() noexcept {
    bool success = true;
    if (end_ != nullptr) {
      const aclrtEvent event = end_;
      end_ = nullptr;
      success = aclrtDestroyEvent(event) == ACL_SUCCESS && success;
    }
    if (start_ != nullptr) {
      const aclrtEvent event = start_;
      start_ = nullptr;
      success = aclrtDestroyEvent(event) == ACL_SUCCESS && success;
    }
    return success;
  }

  aclrtEvent start_ = nullptr;
  aclrtEvent end_ = nullptr;
};

void launch_and_attest(const EngineBuildContext& context,
                       const fs::path& cache_root,
                       const AscendStageArtifact& stage,
                       StageLaunch& launch,
                       aclrtStream stream,
                       const flagdnnBackendBindingV2 bindings[],
                       std::size_t binding_count,
                       void* workspace,
                       std::size_t workspace_size,
                       bool* raw_started,
                       bool* terminal_failure) {
  const CacheSnapshot before =
      scan_cache(cache_root, launch.candidate.entry_point);
  HookCapture capture;
  LaunchHookOwner hook(context, launch.candidate, stream, &capture);
  try {
    RawArgumentPack arguments(stage,
                              launch.candidate,
                              bindings,
                              binding_count,
                              workspace,
                              workspace_size);
    require_locked_configuration(context, terminal_failure);
    ContainedRawLaunch contained_launch{launch.function,
                                        stream,
                                        &launch.candidate,
                                        arguments.data(),
                                        arguments.size(),
                                        raw_started};
    detail::run_with_contained_python_stdout(
        launch_create_raw_with_contained_stdout, &contained_launch);
    require_locked_configuration(context, terminal_failure);
    if (!capture.entered) {
      compilation_failure("libtriton_jit did not invoke the launch-enter hook");
    }
    const CacheSnapshot after =
        scan_cache(cache_root, launch.candidate.entry_point);
    fs::path metadata_path;
    const std::string key = candidate_key(cache_root, context, launch.candidate);
    const RegularFile& metadata =
        select_metadata(before,
                        after,
                        key,
                        launch.candidate.entry_point,
                        &metadata_path);
    validate_metadata(metadata_path,
                      metadata,
                      cache_root,
                      after,
                      launch.candidate,
                      capture.shared_memory);
    require_locked_configuration(context, terminal_failure);
    hook.clear();
  } catch (...) {
    if (*raw_started) {
      latch_process_terminal();
    }
    hook.clear_noexcept();
    throw;
  }
}

void launch_cache_hit(const EngineBuildContext& context,
                      const AscendStageArtifact& stage,
                      const StageLaunch& launch,
                      BuildResources& resources,
                      bool* raw_started,
                      bool* terminal_failure) {
  RawArgumentPack arguments(stage,
                            launch.candidate,
                            resources.bindings(),
                            resources.binding_count(),
                            resources.workspace(),
                            resources.workspace_size());
  require_locked_configuration(context, terminal_failure);
  resources.mark_pending();
  *raw_started = true;
  launch_with_exported_raw_api(*launch.function,
                               resources.stream(),
                               launch.candidate,
                               arguments.data(),
                               arguments.size());
  require_locked_configuration(context, terminal_failure);
}

[[nodiscard]] double measure_candidate(
    const EngineBuildContext& context,
    const fs::path& cache_root,
    const AscendStageArtifact& stage,
    const StageLaunch& launch,
    BuildResources& resources,
    bool* raw_started,
    bool* terminal_failure) {
  const CacheSnapshot before =
      scan_cache(cache_root, launch.candidate.entry_point);
  for (unsigned int index = 0; index < stage.warmup; ++index) {
    launch_cache_hit(context,
                     stage,
                     launch,
                     resources,
                     raw_started,
                     terminal_failure);
  }
  resources.synchronize();

  AutotuneEvents events;
  std::vector<double> samples;
  try {
    samples.reserve(stage.repetitions);
    for (unsigned int index = 0; index < stage.repetitions; ++index) {
      RawArgumentPack arguments(stage,
                                launch.candidate,
                                resources.bindings(),
                                resources.binding_count(),
                                resources.workspace(),
                                resources.workspace_size());
      require_locked_configuration(context, terminal_failure);
      resources.mark_pending();
      check_acl(aclrtRecordEvent(events.start(), resources.stream()),
                "aclrtRecordEvent(autotune start)");
      *raw_started = true;
      launch_with_exported_raw_api(*launch.function,
                                   resources.stream(),
                                   launch.candidate,
                                   arguments.data(),
                                   arguments.size());
      check_acl(aclrtRecordEvent(events.end(), resources.stream()),
                "aclrtRecordEvent(autotune end)");
      require_locked_configuration(context, terminal_failure);
      resources.synchronize();
      float milliseconds = 0.0F;
      check_acl(aclrtEventElapsedTime(
                    &milliseconds, events.start(), events.end()),
                "aclrtEventElapsedTime(autotune)");
      const double microseconds = static_cast<double>(milliseconds) * 1000.0;
      if (!std::isfinite(microseconds) || microseconds < 0.0) {
        compilation_failure(
            "Ascend autotune produced an invalid event sample");
      }
      samples.push_back(microseconds);
    }
    events.release();
  } catch (...) {
    if (resources.synchronize_noexcept()) {
      try {
        events.release();
      } catch (...) {
        latch_process_terminal();
        events.abandon();
      }
    } else {
      events.abandon();
    }
    throw;
  }
  require_locked_configuration(context, terminal_failure);
  const CacheSnapshot after =
      scan_cache(cache_root, launch.candidate.entry_point);
  if (!same_cache_snapshot(before, after)) {
    compilation_failure(
        "Ascend autotune changed the attested private NPU cache tree");
  }
  if (samples.empty()) {
    compilation_failure("Ascend autotune produced no timing samples");
  }
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  if (samples.size() % 2 != 0) {
    return samples[middle];
  }
  return (samples[middle - 1] + samples[middle]) * 0.5;
}

[[nodiscard]] std::vector<std::uint8_t> smoke_candidate(
    const EngineBuildContext& context,
    const fs::path& cache_root,
    const AscendStageArtifact& stage,
    StageLaunch& launch,
    BuildResources& resources,
    bool* raw_started,
    bool* terminal_failure) {
  return detail::run_checked_build_time_prewarm(
      resources, stage, [&]() {
        launch_and_attest(context,
                          cache_root,
                          stage,
                          launch,
                          resources.stream(),
                          resources.bindings(),
                          resources.binding_count(),
                          resources.workspace(),
                          resources.workspace_size(),
                          raw_started,
                          terminal_failure);
      });
}

void validate_candidate_output_consistency(
    const AscendStageArtifact& stage,
    const std::vector<std::uint8_t>& reference,
    const std::vector<std::uint8_t>& candidate) {
  if (stage.kernel_family == KernelFamily::kReduction) {
    return;
  }
  if (stage.kernel_family == KernelFamily::kBatchNorm) {
    ConstBatchNormHostBuffers reference_outputs{};
    ConstBatchNormHostBuffers candidate_outputs{};
    std::size_t reference_offset = 0U;
    std::size_t candidate_offset = 0U;
    for (std::size_t index = 0U; index < reference_outputs.size(); ++index) {
      const std::size_t size =
          stage.arguments[batchnorm_first_output_argument_index() + index].size;
      require(reference_offset <= reference.size() &&
                  size <= reference.size() - reference_offset &&
                  candidate_offset <= candidate.size() &&
                  size <= candidate.size() - candidate_offset,
              "Ascend BatchNorm candidate snapshot size differs",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      reference_outputs[index] = std::span<const std::uint8_t>(
          reference.data() + reference_offset, size);
      candidate_outputs[index] = std::span<const std::uint8_t>(
          candidate.data() + candidate_offset, size);
      reference_offset += size;
      candidate_offset += size;
    }
    require(reference_offset == reference.size() &&
                candidate_offset == candidate.size(),
            "Ascend BatchNorm candidate snapshot size differs",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    try {
      validate_batchnorm_candidate_outputs(
          stage, reference_outputs, candidate_outputs);
    } catch (const std::invalid_argument& error) {
      compilation_failure(error.what());
    }
    return;
  }
  if (stage.kernel_family == KernelFamily::kBatchNormInference) {
    try {
      validate_batchnorm_inference_candidate_outputs(
          stage, reference, candidate);
    } catch (const std::invalid_argument& error) {
      compilation_failure(error.what());
    }
    return;
  }
  if (stage.kernel_family == KernelFamily::kMatMul) {
    try {
      validate_matmul_candidate_outputs(stage, reference, candidate);
    } catch (const std::invalid_argument& error) {
      compilation_failure(error.what());
    }
    return;
  }
  if (stage.kernel_family == KernelFamily::kConvolutionFprop) {
    try {
      validate_convolution_fprop_candidate_outputs(
          stage, reference, candidate);
    } catch (const std::invalid_argument& error) {
      compilation_failure(error.what());
    }
    return;
  }
  if (stage.kernel_family == KernelFamily::kRmsNorm) {
    const std::size_t y_size = stage.arguments[rmsnorm_y_argument_index()].size;
    const std::size_t statistic_size =
        stage.arguments[rmsnorm_inv_variance_argument_index()].size;
    require(y_size <= reference.size() && y_size <= candidate.size() &&
                statistic_size == reference.size() - y_size &&
                statistic_size == candidate.size() - y_size,
            "Ascend RMSNorm candidate snapshot size differs",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    try {
      validate_rmsnorm_candidate_outputs(
          stage,
          {reference.data(), y_size},
          {reference.data() + y_size, statistic_size},
          {candidate.data(), y_size},
          {candidate.data() + y_size, statistic_size});
    } catch (const std::invalid_argument& error) {
      compilation_failure(error.what());
    }
    return;
  }
  if (stage.kernel_family == KernelFamily::kLayerNorm) {
    const std::size_t y_size =
        stage.arguments[layernorm_y_argument_index()].size;
    const std::size_t mean_size =
        stage.arguments[layernorm_mean_argument_index()].size;
    const std::size_t inv_size =
        stage.arguments[layernorm_inv_variance_argument_index()].size;
    require(y_size <= reference.size() && y_size <= candidate.size() &&
                mean_size <= reference.size() - y_size &&
                mean_size <= candidate.size() - y_size &&
                inv_size == reference.size() - y_size - mean_size &&
                inv_size == candidate.size() - y_size - mean_size,
            "Ascend LayerNorm candidate snapshot size differs",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    try {
      validate_layernorm_candidate_outputs(
          stage,
          {reference.data(), y_size},
          {reference.data() + y_size, mean_size},
          {reference.data() + y_size + mean_size, inv_size},
          {candidate.data(), y_size},
          {candidate.data() + y_size, mean_size},
          {candidate.data() + y_size + mean_size, inv_size});
    } catch (const std::invalid_argument& error) {
      compilation_failure(error.what());
    }
    return;
  }
  if (candidate != reference) {
    compilation_failure(
        "Ascend autotune candidates produced different output bytes");
  }
}

[[nodiscard]] std::pair<flagdnnBackendResult_t, std::string>
current_failure(flagdnnBackendResult_t fallback, const char* prefix) {
  try {
    throw;
  } catch (const AscendError& error) {
    return {error.result(), std::string(prefix) + error.what()};
  } catch (const std::bad_alloc&) {
    return {FLAGDNN_BACKEND_RESULT_ALLOC_FAILED,
            std::string(prefix) + "host allocation failed"};
  } catch (const std::exception& error) {
    return {fallback, std::string(prefix) + error.what()};
  } catch (...) {
    return {fallback, std::string(prefix) + "unknown failure"};
  }
}

class LtjExecutionEngine final : public ExecutionEngine {
 public:
  LtjExecutionEngine(EngineBuildContext context,
                     AscendArtifact artifact,
                     std::vector<StageLaunch> launches)
      : context_(std::move(context)),
        artifact_(std::move(artifact)),
        launches_(std::move(launches)) {}

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return artifact_.workspace_size;
  }

  void execute(void* native_stream,
               const flagdnnBackendBindingV2 bindings[],
               std::size_t binding_count,
               void* workspace,
               std::size_t workspace_size) const override {
    ContextGuard context_guard(context_.context);
    aclrtStream stream = static_cast<aclrtStream>(native_stream);
    if (stream == nullptr) {
      check_acl(aclrtCtxGetCurrentDefaultStream(&stream),
                "aclrtCtxGetCurrentDefaultStream");
      require(stream != nullptr,
              "aclrtCtxGetCurrentDefaultStream returned a null stream",
              FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
    }
    std::unique_lock<std::mutex> lock(process_ltj_mutex());
    if (detail::process_domain().terminal_failure_latched()) {
      lock.unlock();
      ensure_process_healthy();
    }

    bool raw_started = false;
    try {
      // The compiler environment, cache directories and toolchain identity are
      // fully validated while constructing this immutable engine. Raw launch
      // does not consult them, so repeating getenv/stat/access scans here
      // would turn each steady-state stage into filesystem control-plane work.
      validate_execution_inputs(
          artifact_, bindings, binding_count, workspace, workspace_size);
      for (const StageLaunch& launch : launches_) {
        require(launch.stage_index < artifact_.stages.size() &&
                    launch.function != nullptr,
                "Ascend executable has an invalid stage launch",
                FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
        const AscendStageArtifact& stage = artifact_.stages[launch.stage_index];
        RawArgumentPack arguments(stage,
                                  launch.candidate,
                                  bindings,
                                  binding_count,
                                  workspace,
                                  workspace_size);
        raw_started = true;
        launch_with_exported_raw_api(
            *launch.function,
            stream,
            launch.candidate,
            arguments.data(),
            arguments.size());
      }
      lock.unlock();
    } catch (...) {
      if (!raw_started) {
        throw;
      }
      latch_process_terminal();
      clear_launch_hooks_noexcept();
      const auto [result, message] = current_failure(
          FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
          "Ascend raw execute failed terminally: ");
      lock.unlock();
      mark_process_terminal(result, message);
      throw AscendError(result, message);
    }
  }

 private:
  EngineBuildContext context_;
  AscendArtifact artifact_;
  std::vector<StageLaunch> launches_;
};

}  // namespace

RawArgumentPack::RawArgumentPack(
    const AscendStageArtifact& stage,
    const LtjNpuRawCandidate& candidate,
    const flagdnnBackendBindingV2 bindings[],
    std::size_t binding_count,
    void* workspace,
    std::size_t workspace_size) {
  require(stage.arguments.size() == candidate.argument_types.size() &&
              !stage.arguments.empty() &&
              stage.arguments.size() <= FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS,
          "Ascend raw argument metadata is inconsistent",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  require(binding_count == 0 || bindings != nullptr,
          "Ascend binding array is null");

  size_ = stage.arguments.size();
  for (std::size_t index = 0; index < stage.arguments.size(); ++index) {
    const ArgumentSource& source = stage.arguments[index];
    require(source.index == index && source.type == candidate.argument_types[index],
            "Ascend raw argument type differs from the selected candidate",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    Slot& slot = slots_[index];
    if (source.source == ArgumentSourceKind::kBinding) {
      slot.pointer = find_binding(bindings, binding_count, source.uid);
      require(reinterpret_cast<std::uintptr_t>(slot.pointer) %
                      source.alignment ==
                  0,
              "Ascend binding does not satisfy artifact alignment");
    } else if (source.source == ArgumentSourceKind::kGraphWorkspace) {
      require(workspace != nullptr && source.workspace_offset <= workspace_size &&
                  source.size <= workspace_size - source.workspace_offset,
              "Ascend Graph workspace argument is out of range");
      const std::uintptr_t address =
          reinterpret_cast<std::uintptr_t>(workspace) +
          source.workspace_offset;
      require(address % source.alignment == 0,
              "Ascend Graph workspace argument is misaligned");
      slot.pointer = reinterpret_cast<void*>(address);
    } else {
      switch (source.type) {
        case RawArgumentType::kI32:
          slot.i32 = std::get<std::int32_t>(source.scalar);
          break;
        case RawArgumentType::kI64:
          slot.i64 = std::get<std::int64_t>(source.scalar);
          break;
        case RawArgumentType::kF32:
          slot.f32 = std::get<float>(source.scalar);
          break;
        case RawArgumentType::kF64:
          slot.f64 = std::get<double>(source.scalar);
          break;
        case RawArgumentType::kPointer:
          throw AscendError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                            "Ascend scalar source has pointer ABI");
      }
    }
  }

  for (std::size_t index = 0; index < stage.arguments.size(); ++index) {
    const ArgumentSource& source = stage.arguments[index];
    Slot& slot = slots_[index];
    switch (source.type) {
      case RawArgumentType::kPointer:
        pointers_[index] = &slot.pointer;
        break;
      case RawArgumentType::kI32:
        pointers_[index] = &slot.i32;
        break;
      case RawArgumentType::kI64:
        pointers_[index] = &slot.i64;
        break;
      case RawArgumentType::kF32:
        pointers_[index] = &slot.f32;
        break;
      case RawArgumentType::kF64:
        pointers_[index] = &slot.f64;
        break;
    }
  }
}

std::unique_ptr<ExecutionEngine> create_libtriton_jit_engine(
    const EngineBuildContext& context, AscendArtifact artifact) {
  if (!context.development_mode) {
    throw AscendError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "Ascend production libtriton_jit execution remains disabled until "
        "the sandbox supervisor, quota/cgroup owner and pre-submit cache "
        "attestation are deployed");
  }
  require(Py_IsInitialized() != 0,
          "development Ascend domain has no initialized embedded Python",
          FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED);
  require(!context.configuration_identity.empty(),
          "development Ascend compiler environment identity is empty",
          FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED);
  const fs::path cache_root =
      validate_private_cache_root(fs::path(context.production_cache_root));

  ContextGuard context_guard(context.context);
  ensure_process_configuration(context);
  std::unique_lock<std::mutex> lock(process_ltj_mutex());
  if (detail::process_domain().terminal_failure_latched()) {
    lock.unlock();
    ensure_process_healthy();
  }

  BuildResources resources;
  bool raw_started = false;
  bool terminal_failure = false;
  try {
    require_locked_configuration(context, &terminal_failure);
    resources.initialize(artifact);
    (void)validate_private_cache_root(cache_root);
    require_locked_configuration(context, &terminal_failure);
    resources.mark_pending();
    std::vector<StageLaunch> launches;
    launches.reserve(artifact.stages.size());
    for (std::size_t stage_index = 0; stage_index < artifact.stages.size();
         ++stage_index) {
      AscendStageArtifact& stage = artifact.stages[stage_index];
      if (!stage.autotune) {
        const auto selected = std::find_if(
            stage.candidates.begin(),
            stage.candidates.end(),
            [&stage](const LtjNpuRawCandidate& candidate) {
              return candidate.candidate_id == stage.selected_candidate;
            });
        if (stage.candidates.size() != 1 ||
            selected == stage.candidates.end()) {
          compilation_failure(
              "Ascend fixed stage does not have exactly one selected candidate");
        }
        validate_source(*selected);
        require_locked_configuration(context, &terminal_failure);
        LtjFunction& function = LtjFunction::get_instance(
            selected->source.string(), selected->entry_point);
        require_locked_configuration(context, &terminal_failure);
        launches.push_back({stage_index, *selected, &function});
        const std::vector<std::uint8_t> fixed_output =
            smoke_candidate(context,
                            cache_root,
                            stage,
                            launches.back(),
                            resources,
                            &raw_started,
                            &terminal_failure);
        /* A second identical launch is the selected-candidate prewarm and must
         * be an in-memory LTJ/cache hit; select_metadata rejects disk changes. */
        const std::vector<std::uint8_t> final_fixed_output =
            detail::run_checked_build_time_prewarm(
                resources, stage, [&]() {
                  resources.mark_pending();
                  launch_and_attest(context,
                                    cache_root,
                                    stage,
                                    launches.back(),
                                    resources.stream(),
                                    resources.bindings(),
                                    resources.binding_count(),
                                    resources.workspace(),
                                    resources.workspace_size(),
                                    &raw_started,
                                    &terminal_failure);
                });
        if (final_fixed_output != fixed_output) {
          compilation_failure(
              "fixed Ascend candidate changed output during final prewarm");
        }
        resources.commit_stage_output(stage, final_fixed_output);
        continue;
      }

      struct MeasuredCandidate {
        StageLaunch launch;
        double median_microseconds = 0.0;
      };
      std::vector<MeasuredCandidate> measured;
      measured.reserve(stage.candidates.size());
      std::vector<std::uint8_t> reference_output;
      for (const LtjNpuRawCandidate& candidate : stage.candidates) {
        validate_source(candidate);
        require_locked_configuration(context, &terminal_failure);
        LtjFunction& function = LtjFunction::get_instance(
            candidate.source.string(), candidate.entry_point);
        require_locked_configuration(context, &terminal_failure);
        StageLaunch launch{stage_index, candidate, &function};
        std::vector<std::uint8_t> output =
            smoke_candidate(context,
                            cache_root,
                            stage,
                            launch,
                            resources,
                            &raw_started,
                            &terminal_failure);
        if (reference_output.empty()) {
          reference_output = output;
        } else {
          validate_candidate_output_consistency(
              stage, reference_output, output);
        }

        /* Establish an exact cache-hit prewarm before collecting event samples. */
        const std::vector<std::uint8_t> cache_hit_output =
            detail::run_checked_build_time_prewarm(
                resources, stage, [&]() {
                  launch_and_attest(context,
                                    cache_root,
                                    stage,
                                    launch,
                                    resources.stream(),
                                    resources.bindings(),
                                    resources.binding_count(),
                                    resources.workspace(),
                                    resources.workspace_size(),
                                    &raw_started,
                                    &terminal_failure);
                });
        validate_candidate_output_consistency(
            stage, reference_output, cache_hit_output);
        const double median = measure_candidate(context,
                                                cache_root,
                                                stage,
                                                launch,
                                                resources,
                                                &raw_started,
                                                &terminal_failure);
        const std::vector<std::uint8_t> measured_output =
            resources.read_stage_output(stage);
        resources.validate_stage_inputs_unchanged(stage);
        resources.validate_stage_output(stage, measured_output);
        validate_candidate_output_consistency(
            stage, reference_output, measured_output);
        measured.push_back({std::move(launch), median});
      }
      if (measured.size() < 2 || reference_output.empty()) {
        compilation_failure(
            "Ascend autotune did not evaluate at least two candidates");
      }
      const auto best = std::min_element(
          measured.begin(),
          measured.end(),
          [](const MeasuredCandidate& left, const MeasuredCandidate& right) {
            return left.median_microseconds < right.median_microseconds;
          });
      require(best != measured.end(),
              "Ascend autotune did not select a candidate",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      StageLaunch selected = best->launch;
      stage.selected_candidate = selected.candidate.candidate_id;

      /* Re-run and attest the selected immutable launch description.  This is
       * the final create-time prewarm and leaves the stage output ready for a
       * dependent stage without putting compilation or tuning in execute(). */
      const std::vector<std::uint8_t> selected_output =
          detail::run_checked_build_time_prewarm(
              resources, stage, [&]() {
                launch_and_attest(context,
                                  cache_root,
                                  stage,
                                  selected,
                                  resources.stream(),
                                  resources.bindings(),
                                  resources.binding_count(),
                                  resources.workspace(),
                                  resources.workspace_size(),
                                  &raw_started,
                                  &terminal_failure);
              });
      validate_candidate_output_consistency(
          stage, reference_output, selected_output);
      resources.commit_stage_output(stage, selected_output);
      std::cerr << "[FLAGDNN_ASCEND_AUTOTUNE] stage=" << stage.stage_id
                << " candidate=" << stage.selected_candidate
                << " median_us=" << best->median_microseconds << '\n';
      launches.push_back(std::move(selected));
    }
    resources.synchronize();
    resources.release();
    require_locked_configuration(context, &terminal_failure);
    auto result = std::make_unique<LtjExecutionEngine>(
        context, std::move(artifact), std::move(launches));
    require_locked_configuration(context, &terminal_failure);
    lock.unlock();
    return result;
  } catch (...) {
    clear_launch_hooks_noexcept();
    const bool resources_released = resources.release_noexcept();
    if (!raw_started && !terminal_failure && resources_released) {
      throw;
    }
    latch_process_terminal();
    auto [result, message] = current_failure(
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        "Ascend development JIT/prewarm failed terminally: ");
    if (!resources_released) {
      result = FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR;
      message += "; development prewarm resources could not be released";
    }
    lock.unlock();
    mark_process_terminal(result, message);
    throw AscendError(result, message);
  }
}

}  // namespace flagdnn::ascend
