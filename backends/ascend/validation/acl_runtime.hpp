/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_ACL_RUNTIME_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_ACL_RUNTIME_HPP_

#include <acl/acl.h>
#include <flagdnn/flagdnn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flagdnn::validation::ascend {

[[nodiscard]] inline std::string acl_error_message(
    aclError status, std::string_view operation) {
  std::string result(operation);
  result += " failed with ACL status ";
  result += std::to_string(status);
  const char* recent = aclGetRecentErrMsg();
  if (recent != nullptr && recent[0] != '\0') {
    result += ": ";
    result += recent;
  }
  return result;
}

inline void check_acl(aclError status, std::string_view operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(acl_error_message(status, operation));
  }
}

[[nodiscard]] inline std::string soc_name() {
  const char* value = aclrtGetSocName();
  if (value == nullptr || value[0] == '\0') {
    throw std::runtime_error("aclrtGetSocName returned no SoC identity");
  }
  return value;
}

[[nodiscard]] inline std::string runtime_package_version() {
  std::array<char, ACL_PKG_VERSION_MAX_SIZE> version{};
  std::array<char, 8> package_name = {'r', 'u', 'n', 't', 'i', 'm', 'e', 0};
  check_acl(aclsysGetVersionStr(package_name.data(), version.data()),
            "aclsysGetVersionStr(runtime)");
  if (version[0] == '\0') {
    throw std::runtime_error(
        "aclsysGetVersionStr(runtime) returned an empty version");
  }
  return version.data();
}

class AclRuntime {
 public:
  explicit AclRuntime(std::int32_t device_ordinal = 0)
      : device_ordinal_(device_ordinal) {
    check_acl(aclInit(nullptr), "aclInit");
    initialized_ = true;
    try {
      check_acl(aclrtSetDevice(device_ordinal_), "aclrtSetDevice");
      device_selected_ = true;
    } catch (...) {
      std::uint64_t ignored = 0;
      (void)aclFinalizeReference(&ignored);
      initialized_ = false;
      throw;
    }
  }

  ~AclRuntime() {
    if (!initialized_) {
      return;
    }
    if (device_selected_) {
      const aclError status = aclrtResetDevice(device_ordinal_);
      if (status != ACL_SUCCESS) {
        std::cerr << "Ascend validation cleanup: "
                  << acl_error_message(status, "aclrtResetDevice") << '\n';
      }
    }
    std::uint64_t remaining = 0;
    const aclError status = aclFinalizeReference(&remaining);
    if (status != ACL_SUCCESS || remaining != 0) {
      std::cerr << "Ascend validation cleanup: "
                << acl_error_message(status, "aclFinalizeReference")
                << ", remaining_ref=" << remaining << '\n';
    }
  }

  AclRuntime(const AclRuntime&) = delete;
  AclRuntime& operator=(const AclRuntime&) = delete;

  void finalize() {
    if (!initialized_) {
      return;
    }
    std::string reset_error;
    if (device_selected_) {
      const aclError status = aclrtResetDevice(device_ordinal_);
      device_selected_ = false;
      if (status != ACL_SUCCESS) {
        reset_error = acl_error_message(status, "aclrtResetDevice");
      }
    }

    std::uint64_t remaining = 0;
    const aclError finalize_status = aclFinalizeReference(&remaining);
    initialized_ = false;
    if (finalize_status != ACL_SUCCESS) {
      throw std::runtime_error(
          acl_error_message(finalize_status, "aclFinalizeReference"));
    }
    if (!reset_error.empty()) {
      throw std::runtime_error(std::move(reset_error));
    }
    if (remaining != 0) {
      throw std::runtime_error(
          "aclFinalizeReference left a non-zero reference count: " +
          std::to_string(remaining));
    }
  }

 private:
  std::int32_t device_ordinal_ = 0;
  bool initialized_ = false;
  bool device_selected_ = false;
};

class Stream {
 public:
  Stream() { check_acl(aclrtCreateStream(&value_), "aclrtCreateStream"); }

  ~Stream() {
    if (value_ != nullptr) {
      const aclError status = aclrtDestroyStream(value_);
      if (status != ACL_SUCCESS) {
        std::cerr << "Ascend validation cleanup: "
                  << acl_error_message(status, "aclrtDestroyStream") << '\n';
      }
    }
  }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  [[nodiscard]] aclrtStream get() const noexcept { return value_; }
  [[nodiscard]] flagdnnStream_t opaque() const noexcept {
    return reinterpret_cast<flagdnnStream_t>(value_);
  }

  void synchronize() const {
    check_acl(aclrtSynchronizeStream(value_), "aclrtSynchronizeStream");
  }

 private:
  aclrtStream value_ = nullptr;
};

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0) {
      check_acl(aclrtMalloc(&value_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc");
    }
  }

  ~DeviceBuffer() {
    if (value_ != nullptr) {
      const aclError status = aclrtFree(value_);
      if (status != ACL_SUCCESS) {
        std::cerr << "Ascend validation cleanup: "
                  << acl_error_message(status, "aclrtFree") << '\n';
      }
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }
  [[nodiscard]] void* opaque() const noexcept { return value_; }

  [[nodiscard]] void* opaque_at(std::size_t byte_offset) const {
    if (value_ == nullptr || byte_offset >= bytes_) {
      throw std::invalid_argument("device pointer offset exceeds buffer");
    }
    return static_cast<void*>(
        static_cast<std::uint8_t*>(value_) + byte_offset);
  }

  void copy_from_host_at(const void* source,
                         std::size_t bytes,
                         std::size_t byte_offset,
                         aclrtStream stream) const {
    if (source == nullptr || byte_offset > bytes_ ||
        bytes > bytes_ - byte_offset) {
      throw std::invalid_argument("host-to-device copy exceeds buffer");
    }
    check_acl(aclrtMemcpyAsync(static_cast<std::uint8_t*>(value_) + byte_offset,
                               bytes_ - byte_offset,
                               source,
                               bytes,
                               ACL_MEMCPY_HOST_TO_DEVICE,
                               stream),
              "aclrtMemcpyAsync(host-to-device)");
  }

  void copy_to_host_at(void* destination,
                       std::size_t bytes,
                       std::size_t byte_offset,
                       aclrtStream stream) const {
    if (destination == nullptr || byte_offset > bytes_ ||
        bytes > bytes_ - byte_offset) {
      throw std::invalid_argument("device-to-host copy exceeds buffer");
    }
    check_acl(aclrtMemcpyAsync(destination,
                               bytes,
                               static_cast<const std::uint8_t*>(value_) +
                                   byte_offset,
                               bytes,
                               ACL_MEMCPY_DEVICE_TO_HOST,
                               stream),
              "aclrtMemcpyAsync(device-to-host)");
  }

 private:
  void* value_ = nullptr;
  std::size_t bytes_ = 0;
};

struct TimingSample {
  double stream_us = 0.0;
  double submit_us = 0.0;
  double end_to_end_us = 0.0;
};

class EventTimer {
 public:
  EventTimer() {
    check_acl(aclrtCreateEventExWithFlag(&start_, ACL_EVENT_TIME_LINE),
              "aclrtCreateEventExWithFlag(start)");
    try {
      check_acl(aclrtCreateEventExWithFlag(&end_, ACL_EVENT_TIME_LINE),
                "aclrtCreateEventExWithFlag(end)");
    } catch (...) {
      (void)aclrtDestroyEvent(start_);
      start_ = nullptr;
      throw;
    }
  }

  ~EventTimer() {
    if (end_ != nullptr) {
      const aclError status = aclrtDestroyEvent(end_);
      if (status != ACL_SUCCESS) {
        std::cerr << "Ascend validation cleanup: "
                  << acl_error_message(status, "aclrtDestroyEvent(end)")
                  << '\n';
      }
    }
    if (start_ != nullptr) {
      const aclError status = aclrtDestroyEvent(start_);
      if (status != ACL_SUCCESS) {
        std::cerr << "Ascend validation cleanup: "
                  << acl_error_message(status, "aclrtDestroyEvent(start)")
                  << '\n';
      }
    }
  }

  EventTimer(const EventTimer&) = delete;
  EventTimer& operator=(const EventTimer&) = delete;

  template <typename Function>
  [[nodiscard]] TimingSample measure(aclrtStream stream,
                                     int iterations,
                                     Function&& execute) {
    if (stream == nullptr || iterations <= 0) {
      throw std::invalid_argument("ACL benchmark sample is invalid");
    }
    check_acl(aclrtRecordEvent(start_, stream), "aclrtRecordEvent(start)");
    const auto submit_begin = std::chrono::steady_clock::now();
    for (int index = 0; index < iterations; ++index) {
      execute();
    }
    const auto submit_end = std::chrono::steady_clock::now();
    check_acl(aclrtRecordEvent(end_, stream), "aclrtRecordEvent(end)");
    check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(sample)");
    const auto completed = std::chrono::steady_clock::now();

    float milliseconds = 0.0F;
    check_acl(aclrtEventElapsedTime(&milliseconds, start_, end_),
              "aclrtEventElapsedTime");
    const double divisor = static_cast<double>(iterations);
    TimingSample result;
    result.stream_us = static_cast<double>(milliseconds) * 1000.0 / divisor;
    result.submit_us =
        std::chrono::duration<double, std::micro>(submit_end - submit_begin)
            .count() /
        divisor;
    result.end_to_end_us =
        std::chrono::duration<double, std::micro>(completed - submit_begin)
            .count() /
        divisor;
    return result;
  }

 private:
  aclrtEvent start_ = nullptr;
  aclrtEvent end_ = nullptr;
};

}  // namespace flagdnn::validation::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_ACL_RUNTIME_HPP_
