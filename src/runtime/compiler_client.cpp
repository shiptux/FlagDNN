/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "runtime/compiler_client.hpp"

#include "error.hpp"
#include "runtime/context.hpp"
#include "runtime/json.hpp"
#include "runtime/sha256.hpp"

#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

extern char **environ;

namespace flagdnn::native {
namespace {

std::atomic<std::uint64_t> identity_temporary_counter{0};

constexpr int kIdentityTemporaryFailure = 75;
constexpr std::size_t kIdentityQueryAttempts = 3;

class TemporaryIdentityFailure final : public std::exception {
public:
  [[nodiscard]] const char *what() const noexcept override {
    return "compiler identity dependencies changed during discovery";
  }
};

std::chrono::seconds compiler_timeout() {
  constexpr std::uint64_t default_seconds = 1800;
  constexpr std::uint64_t maximum_seconds = 86400;
  const char *configured = std::getenv("FLAGDNN_COMPILER_TIMEOUT_SECONDS");
  if (configured == nullptr || configured[0] == '\0') {
    return std::chrono::seconds(default_seconds);
  }
  const std::string_view value(configured);
  std::uint64_t seconds = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), seconds);
  if (error != std::errc{} || end != value.data() + value.size() ||
      seconds == 0 || seconds > maximum_seconds) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "FLAGDNN_COMPILER_TIMEOUT_SECONDS must be an integer in [1, 86400]");
  }
  return std::chrono::seconds(seconds);
}

bool is_sha256(std::string_view value) {
  if (value.size() != 64) {
    return false;
  }
  for (const char character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

class TemporaryFile {
public:
  explicit TemporaryFile(std::filesystem::path path) : path_(std::move(path)) {}

  ~TemporaryFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

private:
  std::filesystem::path path_;
};

void run_compiler_process(const RuntimeContext &context,
                          const std::vector<std::string> &owned_arguments,
                          std::string_view action,
                          bool retry_temporary_identity_failure = false) {
  if (context.compiler_executable().empty() || context.compiler().empty()) {
    throw ApiError(
        FLAGDNN_STATUS_COMPILATION_FAILED,
        "external compiler is not configured; call flagdnnSetCompilerConfig");
  }
  if (owned_arguments.size() < 2 ||
      owned_arguments[0] != context.compiler_executable() ||
      owned_arguments[1] != context.compiler()) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "external compiler argument construction is invalid");
  }

  std::vector<char *> arguments;
  arguments.reserve(owned_arguments.size() + 1);
  for (const std::string &argument : owned_arguments) {
    arguments.push_back(const_cast<char *>(argument.c_str()));
  }
  arguments.push_back(nullptr);

  // Copy the inherited environment into owned strings before spawning. This
  // avoids retaining string_views or other pointers into `environ` while the
  // child runs and permits independent compiler processes to run concurrently.
  std::vector<std::string> owned_environment;
  pid_t child = -1;
  int spawn_result = 0;
  std::chrono::seconds timeout;
  posix_spawnattr_t spawn_attributes;
  int attribute_result = posix_spawnattr_init(&spawn_attributes);
  if (attribute_result != 0) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "cannot initialize external compiler process attributes: " +
                       std::string(std::strerror(attribute_result)));
  }
  attribute_result =
      posix_spawnattr_setflags(&spawn_attributes, POSIX_SPAWN_SETPGROUP);
  if (attribute_result == 0) {
    // A process-group value of zero creates a group whose ID is the child PID.
    // This lets timeout handling terminate Python plus ptxas/other descendants.
    attribute_result = posix_spawnattr_setpgroup(&spawn_attributes, 0);
  }
  if (attribute_result != 0) {
    (void)posix_spawnattr_destroy(&spawn_attributes);
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "cannot configure external compiler process group: " +
                       std::string(std::strerror(attribute_result)));
  }
  {
    const std::shared_lock environment_lock(process_environment_mutex());
    timeout = compiler_timeout();
    if (environ != nullptr) {
      for (char **entry = environ; *entry != nullptr; ++entry) {
        owned_environment.emplace_back(*entry);
      }
    }
    std::vector<char *> environment;
    environment.reserve(owned_environment.size() + 1);
    for (std::string &entry : owned_environment) {
      environment.push_back(entry.data());
    }
    environment.push_back(nullptr);

    spawn_result =
        posix_spawnp(&child, arguments[0], nullptr, &spawn_attributes,
                     arguments.data(), environment.data());
  }
  (void)posix_spawnattr_destroy(&spawn_attributes);
  if (spawn_result != 0) {
    const std::string message = "cannot start external compiler for " +
                                std::string(action) + ": " +
                                std::string(std::strerror(spawn_result));
    if (spawn_result == ENOENT) {
      throw CompilerExecutableUnavailable(message);
    }
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED, message);
  }

  const auto deadline = std::chrono::steady_clock::now() + timeout;
  int status = 0;
  pid_t wait_result = -1;
  for (;;) {
    wait_result = waitpid(child, &status, WNOHANG);
    if (wait_result == child) {
      break;
    }
    if (wait_result < 0 && errno != EINTR) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      (void)kill(-child, SIGKILL);
      do {
        wait_result = waitpid(child, &status, 0);
      } while (wait_result < 0 && errno == EINTR);
      throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                     "external compiler " + std::string(action) +
                         " timed out after " + std::to_string(timeout.count()) +
                         " seconds");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (wait_result != child) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "waitpid failed for external compiler: " +
                       std::string(std::strerror(errno)));
  }
  if (retry_temporary_identity_failure && WIFEXITED(status) &&
      WEXITSTATUS(status) == kIdentityTemporaryFailure) {
    throw TemporaryIdentityFailure();
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::ostringstream message;
    message << "external compiler " << action << " failed";
    if (WIFEXITED(status)) {
      message << " with exit code " << WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      message << " with signal " << WTERMSIG(status);
    }
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED, message.str());
  }
}

struct CompilerIdentityResponse {
  struct DependencySnapshot {
    std::filesystem::path path;
    std::string fingerprint;
    std::string content_sha256;
  };

  std::string digest;
  std::vector<std::filesystem::path> dependencies;
  std::vector<DependencySnapshot> snapshots;
  bool has_snapshots = false;
  bool dependencies_complete = false;
};

CompilerIdentityResponse
read_compiler_identity(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "external compiler did not produce its identity");
  }
  std::string value((std::istreambuf_iterator<char>(input)),
                    std::istreambuf_iterator<char>());
  constexpr std::size_t maximum_identity_bytes = 8U * 1024U * 1024U;
  if (input.bad() || value.size() > maximum_identity_bytes) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "compiler identity file is invalid");
  }
  const std::size_t newline = value.find('\n');
  std::string digest = value.substr(0, newline);
  if (!digest.empty() && digest.back() == '\r') {
    digest.pop_back();
  }
  if (!is_sha256(digest)) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "external compiler returned an invalid identity");
  }

  CompilerIdentityResponse result;
  result.digest = std::move(digest);
  if (newline == std::string::npos) {
    return result;
  }
  const std::string_view metadata(value.data() + newline + 1,
                                  value.size() - newline - 1);
  if (std::all_of(metadata.begin(), metadata.end(), [](char character) {
        return std::isspace(static_cast<unsigned char>(character)) != 0;
      })) {
    return result;
  }
  try {
    const auto root = json::parse(metadata);
    if (root.at("schema_version").as_int() != 1) {
      throw std::runtime_error("unsupported dependency schema");
    }
    const auto &files = root.at("files").as_array();
    constexpr std::size_t maximum_dependency_count = 65536;
    constexpr std::size_t maximum_dependency_path_bytes = 65536;
    if (files.size() > maximum_dependency_count) {
      throw std::runtime_error("too many dependency paths");
    }
    result.dependencies.reserve(files.size());
    for (const auto &file : files) {
      const std::string &text = file.as_string();
      const std::filesystem::path dependency(text);
      if (text.empty() || text.size() > maximum_dependency_path_bytes ||
          !dependency.is_absolute()) {
        throw std::runtime_error(
            "dependency paths must be nonempty absolute paths");
      }
      result.dependencies.push_back(dependency.lexically_normal());
    }
    std::sort(result.dependencies.begin(), result.dependencies.end());
    result.dependencies.erase(
        std::unique(result.dependencies.begin(), result.dependencies.end()),
        result.dependencies.end());

    const auto &object = root.as_object();
    const auto completeness = object.find("dependencies_complete");
    if (completeness != object.end()) {
      result.dependencies_complete = completeness->second.as_bool();
    }
    const auto snapshot_schema = object.find("snapshot_schema_version");
    const auto snapshots = object.find("snapshots");
    if ((snapshot_schema == object.end()) != (snapshots == object.end())) {
      throw std::runtime_error(
          "snapshot_schema_version and snapshots must appear together");
    }
    if (snapshots != object.end()) {
      if (snapshot_schema->second.as_int() != 1) {
        throw std::runtime_error("unsupported dependency snapshot schema");
      }
      const auto &entries = snapshots->second.as_array();
      if (entries.size() > maximum_dependency_count) {
        throw std::runtime_error("too many dependency snapshots");
      }
      result.snapshots.reserve(entries.size());
      for (const auto &entry : entries) {
        const std::string &text = entry.at("path").as_string();
        const std::string &fingerprint = entry.at("fingerprint").as_string();
        const std::filesystem::path dependency(text);
        if (text.empty() || text.size() > maximum_dependency_path_bytes ||
            !dependency.is_absolute() || !is_sha256(fingerprint)) {
          throw std::runtime_error(
              "dependency snapshots contain an invalid path or fingerprint");
        }
        std::string content_sha256;
        const auto &snapshot_object = entry.as_object();
        const auto content = snapshot_object.find("content_sha256");
        if (content != snapshot_object.end()) {
          content_sha256 = content->second.as_string();
          if (!is_sha256(content_sha256)) {
            throw std::runtime_error(
                "dependency snapshot content hash is invalid");
          }
        }
        result.snapshots.push_back(
            {dependency.lexically_normal(), fingerprint, content_sha256});
      }
      std::sort(result.snapshots.begin(), result.snapshots.end(),
                [](const auto &left, const auto &right) {
                  return left.path < right.path;
                });
      if (std::adjacent_find(result.snapshots.begin(), result.snapshots.end(),
                             [](const auto &left, const auto &right) {
                               return left.path == right.path;
                             }) != result.snapshots.end()) {
        throw std::runtime_error(
            "dependency snapshots contain duplicate paths");
      }
      if (result.snapshots.size() != result.dependencies.size()) {
        throw std::runtime_error(
            "dependency snapshots do not match dependency paths");
      }
      for (std::size_t index = 0; index < result.dependencies.size(); ++index) {
        if (result.snapshots[index].path != result.dependencies[index]) {
          throw std::runtime_error(
              "dependency snapshots do not match dependency paths");
        }
      }
      result.has_snapshots = true;
    }
    if (result.dependencies_complete && !result.has_snapshots) {
      throw std::runtime_error(
          "complete dependencies require dependency snapshots");
    }
  } catch (const std::exception &error) {
    throw ApiError(
        FLAGDNN_STATUS_COMPILATION_FAILED,
        "external compiler returned invalid identity dependencies: " +
            std::string(error.what()));
  }
  return result;
}

void append_stat_record(std::string &state, const struct stat &status) {
  state += std::to_string(static_cast<std::uintmax_t>(status.st_dev));
  state.push_back(':');
  state += std::to_string(static_cast<std::uintmax_t>(status.st_ino));
  state.push_back(':');
  state += std::to_string(static_cast<std::uintmax_t>(status.st_mode));
  state.push_back(':');
  state += std::to_string(static_cast<std::uintmax_t>(status.st_size));
  state.push_back(':');
#if defined(__APPLE__)
  const auto &modified = status.st_mtimespec;
  const auto &changed = status.st_ctimespec;
#else
  const auto &modified = status.st_mtim;
  const auto &changed = status.st_ctim;
#endif
  state += std::to_string(modified.tv_sec);
  state.push_back(':');
  state += std::to_string(modified.tv_nsec);
  state.push_back(':');
  state += std::to_string(changed.tv_sec);
  state.push_back(':');
  state += std::to_string(changed.tv_nsec);
}

std::string dependency_fingerprint(const std::filesystem::path &path) {
  std::string state("flagdnn-dependency-state-v1\0", 28);
  struct stat link_status {};
  if (::lstat(path.c_str(), &link_status) != 0) {
    const int error = errno;
    state += "lstat-error:";
    state += std::to_string(error);
    state.push_back('\0');
    return sha256(state);
  }
  state += "lstat:";
  append_stat_record(state, link_status);
  state.push_back('\0');

  if (S_ISLNK(link_status.st_mode)) {
    std::vector<char> link_value(256);
    for (;;) {
      const ssize_t size =
          ::readlink(path.c_str(), link_value.data(), link_value.size());
      if (size < 0) {
        const int error = errno;
        state += "link-error:";
        state += std::to_string(error);
        state.push_back('\0');
        break;
      }
      if (static_cast<std::size_t>(size) < link_value.size()) {
        state += "link:";
        state.append(link_value.data(), static_cast<std::size_t>(size));
        state.push_back('\0');
        break;
      }
      if (link_value.size() >= 65536U) {
        state += "link-error:";
        state += std::to_string(ENAMETOOLONG);
        state.push_back('\0');
        break;
      }
      link_value.resize(link_value.size() * 2U);
    }
  } else {
    state.append("link:\0", 6);
  }

  struct stat status {};
  if (::stat(path.c_str(), &status) != 0) {
    const int error = errno;
    state += "stat-error:";
    state += std::to_string(error);
    state.push_back('\0');
  } else {
    state += "stat:";
    append_stat_record(state, status);
    state.push_back('\0');
  }
  return sha256(state);
}

void append_file_identity(std::string &snapshot,
                          const std::filesystem::path &path) {
  snapshot += path.string();
  snapshot.push_back('\0');
  snapshot += dependency_fingerprint(path);
  snapshot.push_back('\0');
}

std::string compiler_dependency_snapshot(
    const std::vector<std::filesystem::path> &dependencies) {
  std::string snapshot;
  for (const auto &dependency : dependencies) {
    append_file_identity(snapshot, dependency);
  }
  return snapshot;
}

std::string
reported_dependency_snapshot(const CompilerIdentityResponse &response) {
  std::string snapshot;
  for (const auto &entry : response.snapshots) {
    snapshot += entry.path.string();
    snapshot.push_back('\0');
    snapshot += entry.fingerprint;
    snapshot.push_back('\0');
  }
  return snapshot;
}

bool reported_dependency_contents_match(
    const CompilerIdentityResponse &response) {
  for (const auto &entry : response.snapshots) {
    if (entry.content_sha256.empty()) {
      continue;
    }
    try {
      if (sha256_file(entry.path) != entry.content_sha256) {
        return false;
      }
    } catch (const std::exception &) {
      return false;
    }
  }
  return true;
}

std::string compiler_identity_snapshot(const RuntimeContext &context) {
  const std::shared_lock environment_lock(process_environment_mutex());
  std::string snapshot;
  snapshot.reserve(4096);
  snapshot += context.compiler_executable();
  snapshot.push_back('\0');
  snapshot += context.compiler();
  snapshot.push_back('\0');
  snapshot += context.backend_name();
  snapshot.push_back('\0');
  snapshot += context.target_fingerprint();
  snapshot.push_back('\0');
  snapshot += context.execution_engine();
  snapshot.push_back('\0');

  if (context.compiler_executable().find('/') != std::string::npos) {
    append_file_identity(snapshot, context.compiler_executable());
  }
  append_file_identity(snapshot, context.compiler());

  std::error_code current_directory_error;
  const std::filesystem::path current_directory =
      std::filesystem::current_path(current_directory_error);
  snapshot += current_directory_error
                  ? "cwd-error:" + current_directory_error.message()
                  : current_directory.string();
  snapshot.push_back('\0');

  // The identity subprocess inherits the complete environment. Sorting it
  // makes the memo key stable. Provider-reported files are tracked separately
  // so in-place kernel/registry/tuning/JIT changes also invalidate the memo.
  std::vector<std::string> environment;
  if (environ != nullptr) {
    for (char **entry = environ; *entry != nullptr; ++entry) {
      environment.emplace_back(*entry);
    }
  }
  std::sort(environment.begin(), environment.end());
  for (const std::string &entry : environment) {
    snapshot.append(entry);
    snapshot.push_back('\0');
  }
  return snapshot;
}

} // namespace

std::string
query_compiler_identity(RuntimeContext &context,
                        const std::filesystem::path &graph_cache_directory,
                        bool force_refresh) {
  // prepare_artifact_package() owns compiler_configuration_mutex_ for the
  // complete identity/cache/compile transaction. Keeping the lock at that
  // boundary also makes set_compiler() atomic with respect to every compiler
  // configuration read below.
  const std::string snapshot = compiler_identity_snapshot(context);
  if (!force_refresh && !context.compiler_identity_.empty() &&
      context.compiler_identity_snapshot_ == snapshot &&
      context.compiler_identity_dependencies_snapshot_ ==
          compiler_dependency_snapshot(
              context.compiler_identity_dependencies_)) {
    return context.compiler_identity_;
  }
  for (std::size_t attempt = 0; attempt < kIdentityQueryAttempts; ++attempt) {
    const std::string snapshot_before = compiler_identity_snapshot(context);
    const std::uint64_t serial = identity_temporary_counter.fetch_add(1);
    const std::filesystem::path identity_output =
        graph_cache_directory / (".identity.tmp." + std::to_string(getpid()) +
                                 "." + std::to_string(serial));
    TemporaryFile cleanup(identity_output);

    try {
      run_compiler_process(context,
                           {context.compiler_executable(), context.compiler(),
                            "--identify", "--backend", context.backend_name(),
                            "--target", context.target_fingerprint(),
                            "--execution-engine", context.execution_engine(),
                            "--identity-output", identity_output.string(),
                            "--quiet"},
                           "identity query", true);
    } catch (const TemporaryIdentityFailure &) {
      continue;
    }

    CompilerIdentityResponse response = read_compiler_identity(identity_output);
    const std::string dependencies_before =
        compiler_dependency_snapshot(response.dependencies);
    if (response.has_snapshots &&
        dependencies_before != reported_dependency_snapshot(response)) {
      continue;
    }
    if (response.has_snapshots &&
        !reported_dependency_contents_match(response)) {
      continue;
    }
    const std::string snapshot_after = compiler_identity_snapshot(context);
    const std::string dependencies_after =
        compiler_dependency_snapshot(response.dependencies);
    if (snapshot_before != snapshot_after ||
        dependencies_before != dependencies_after) {
      continue;
    }

    if (response.dependencies_complete && response.has_snapshots) {
      context.compiler_identity_snapshot_ = snapshot_after;
      context.compiler_identity_dependencies_ =
          std::move(response.dependencies);
      context.compiler_identity_dependencies_snapshot_ =
          std::move(dependencies_after);
      context.compiler_identity_ = std::move(response.digest);
      return context.compiler_identity_;
    }

    // Digest-only and legacy/incomplete dependency responses stay supported,
    // but must retain the historical query-per-build behavior. Otherwise a
    // provider that hashes unreported resources could return a stale memo.
    context.compiler_identity_snapshot_.clear();
    context.compiler_identity_dependencies_.clear();
    context.compiler_identity_dependencies_snapshot_.clear();
    context.compiler_identity_.clear();
    return response.digest;
  }
  throw ApiError(
      FLAGDNN_STATUS_COMPILATION_FAILED,
      "external compiler identity dependencies did not stabilize after " +
          std::to_string(kIdentityQueryAttempts) + " attempts");
}

void compile_external_artifact(const RuntimeContext &context,
                               const std::filesystem::path &request,
                               const std::filesystem::path &output_directory) {
  run_compiler_process(context,
                       {context.compiler_executable(), context.compiler(),
                        "--request", request.string(), "--output-dir",
                        output_directory.string(), "--execution-engine",
                        context.execution_engine(), "--quiet"},
                       "compile");
}

} // namespace flagdnn::native
