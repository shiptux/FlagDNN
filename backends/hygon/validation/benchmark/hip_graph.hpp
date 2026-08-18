/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_HIP_GRAPH_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_HIP_GRAPH_HPP_

#include "hip_driver.hpp"

#include <stdexcept>
#include <utility>

namespace flagdnn::benchmarking {

class CapturedExecutionBatch final {
public:
  template <typename Function>
  CapturedExecutionBatch(hipStream_t stream, int execution_count,
                         Function &&execute)
      : execution_count_(execution_count) {
    if (execution_count_ <= 0) {
      throw std::invalid_argument("captured execution count must be positive");
    }

    validation::hygon::check_hip(
        hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed),
        "hipStreamBeginCapture");
    try {
      for (int index = 0; index < execution_count_; ++index) {
        execute();
      }
    } catch (...) {
      hipGraph_t abandoned_graph = nullptr;
      if (hipStreamEndCapture(stream, &abandoned_graph) == hipSuccess &&
          abandoned_graph != nullptr) {
        (void)hipGraphDestroy(abandoned_graph);
      }
      throw;
    }

    validation::hygon::check_hip(hipStreamEndCapture(stream, &graph_),
                                 "hipStreamEndCapture");
    try {
      validation::hygon::check_hip(
          hipGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0),
          "hipGraphInstantiate");
    } catch (...) {
      (void)hipGraphDestroy(graph_);
      graph_ = nullptr;
      throw;
    }
  }

  CapturedExecutionBatch(const CapturedExecutionBatch &) = delete;
  CapturedExecutionBatch &operator=(const CapturedExecutionBatch &) = delete;

  ~CapturedExecutionBatch() {
    if (executable_ != nullptr) {
      (void)hipGraphExecDestroy(executable_);
    }
    if (graph_ != nullptr) {
      (void)hipGraphDestroy(graph_);
    }
  }

  void launch(hipStream_t stream) const {
    validation::hygon::check_hip(hipGraphLaunch(executable_, stream),
                                 "hipGraphLaunch");
  }

  [[nodiscard]] int execution_count() const noexcept {
    return execution_count_;
  }

private:
  hipGraph_t graph_ = nullptr;
  hipGraphExec_t executable_ = nullptr;
  int execution_count_ = 0;
};

} // namespace flagdnn::benchmarking

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_HIP_GRAPH_HPP_
