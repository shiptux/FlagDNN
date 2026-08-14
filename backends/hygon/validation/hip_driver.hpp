/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_HIP_DRIVER_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_HIP_DRIVER_HPP_

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flagdnn::validation::hygon {

[[nodiscard]] constexpr std::string_view
hip_error_text_or(const char *text, std::string_view fallback) noexcept {
  return text != nullptr ? std::string_view{text} : fallback;
}

static_assert(hip_error_text_or(nullptr, "unavailable") == "unavailable");
static_assert(hip_error_text_or("known", "unavailable") == "known");

[[nodiscard]] inline std::string describe_hip_error(hipError_t status) {
  return std::string(hip_error_text_or(hipGetErrorName(status),
                                       "HIP error name unavailable")) +
         ": " +
         std::string(hip_error_text_or(hipGetErrorString(status),
                                       "HIP error description unavailable"));
}

inline void check_hip(hipError_t status, const char *operation) {
  if (status == hipSuccess) {
    return;
  }
  throw std::runtime_error(std::string(operation) +
                           " failed: " + describe_hip_error(status));
}

class DeviceGuard {
public:
  explicit DeviceGuard(int ordinal = 0) {
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
  [[nodiscard]] void *opaque() const noexcept {
    return reinterpret_cast<void *>(value_);
  }

  void synchronize() const {
    check_hip(hipStreamSynchronize(value_), "hipStreamSynchronize");
  }

private:
  hipStream_t value_ = nullptr;
};

class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t bytes)
      : bytes_(std::max<std::size_t>(bytes, 1)) {
    check_hip(hipMalloc(&value_, bytes_), "hipMalloc");
  }

  ~DeviceBuffer() {
    if (value_ != nullptr) {
      (void)hipFree(value_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept
      : value_(std::exchange(other.value_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)) {}
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      if (value_ != nullptr) {
        (void)hipFree(value_);
      }
      value_ = std::exchange(other.value_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }

  [[nodiscard]] void *opaque() const noexcept { return value_; }
  [[nodiscard]] void *opaque_at(std::size_t byte_offset) const {
    if (byte_offset >= bytes_) {
      throw std::invalid_argument("device pointer offset exceeds buffer");
    }
    return static_cast<void *>(static_cast<std::uint8_t *>(value_) +
                               byte_offset);
  }

  void copy_from_host(const void *source, std::size_t bytes,
                      hipStream_t stream) const {
    copy_from_host_at(source, bytes, 0, stream);
  }

  void copy_from_host_at(const void *source, std::size_t bytes,
                         std::size_t byte_offset, hipStream_t stream) const {
    if (byte_offset > bytes_ || bytes > bytes_ - byte_offset) {
      throw std::invalid_argument("host-to-device copy exceeds buffer");
    }
    check_hip(hipMemcpyAsync(static_cast<std::uint8_t *>(value_) + byte_offset,
                             source, bytes, hipMemcpyHostToDevice, stream),
              "hipMemcpyAsync(H2D)");
  }

  void copy_to_host(void *destination, std::size_t bytes,
                    hipStream_t stream) const {
    copy_to_host_at(destination, bytes, 0, stream);
  }

  void copy_to_host_at(void *destination, std::size_t bytes,
                       std::size_t byte_offset, hipStream_t stream) const {
    if (byte_offset > bytes_ || bytes > bytes_ - byte_offset) {
      throw std::invalid_argument("device-to-host copy exceeds buffer");
    }
    check_hip(hipMemcpyAsync(destination,
                             static_cast<std::uint8_t *>(value_) + byte_offset,
                             bytes, hipMemcpyDeviceToHost, stream),
              "hipMemcpyAsync(D2H)");
  }

private:
  void *value_ = nullptr;
  std::size_t bytes_ = 0;
};

class EventTimer {
public:
  EventTimer() {
    check_hip(hipEventCreate(&start_), "hipEventCreate(start)");
    try {
      check_hip(hipEventCreate(&stop_), "hipEventCreate(stop)");
    } catch (...) {
      (void)hipEventDestroy(start_);
      throw;
    }
  }

  ~EventTimer() {
    if (stop_ != nullptr) {
      (void)hipEventDestroy(stop_);
    }
    if (start_ != nullptr) {
      (void)hipEventDestroy(start_);
    }
  }

  EventTimer(const EventTimer &) = delete;
  EventTimer &operator=(const EventTimer &) = delete;

  template <typename Function>
  double measure_microseconds(hipStream_t stream, int iterations,
                              Function &&function) {
    if (iterations <= 0) {
      throw std::invalid_argument("benchmark iterations must be positive");
    }
    check_hip(hipEventRecord(start_, stream), "hipEventRecord(start)");
    for (int index = 0; index < iterations; ++index) {
      function();
    }
    check_hip(hipEventRecord(stop_, stream), "hipEventRecord(stop)");
    check_hip(hipEventSynchronize(stop_), "hipEventSynchronize(stop)");
    float milliseconds = 0.0F;
    check_hip(hipEventElapsedTime(&milliseconds, start_, stop_),
              "hipEventElapsedTime");
    return static_cast<double>(milliseconds) * 1000.0 /
           static_cast<double>(iterations);
  }

private:
  hipEvent_t start_ = nullptr;
  hipEvent_t stop_ = nullptr;
};

} // namespace flagdnn::validation::hygon

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_HIP_DRIVER_HPP_
