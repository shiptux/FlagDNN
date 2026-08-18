/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ENGINES_LIBTRITON_JIT_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ENGINES_LIBTRITON_JIT_HPP_

#include "backends/ascend/artifact.hpp"
#include "backends/ascend/engines/engine.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace flagdnn::ascend {

/*
 * Stable host-side slots for libtriton_jit's raw NPU ABI. Each table entry is
 * the address of a typed host slot; pointer entries are not device addresses
 * cast directly to void*.
 */
class RawArgumentPack {
 public:
  RawArgumentPack(const AscendStageArtifact& stage,
                  const LtjNpuRawCandidate& candidate,
                  const flagdnnBackendBindingV2 bindings[],
                  std::size_t binding_count,
                  void* workspace,
                  std::size_t workspace_size);

  [[nodiscard]] void** data() noexcept { return pointers_.data(); }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  struct Slot {
    void* pointer;
    std::int32_t i32;
    std::int64_t i64;
    float f32;
    double f64;
  };

  // Only the prefix described by size_ is observable. Leaving the unused
  // tail uninitialized avoids clearing roughly 160 KiB on every stage launch.
  std::array<Slot, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS> slots_;
  std::array<void*, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS> pointers_;
  std::size_t size_ = 0;
};

[[nodiscard]] std::unique_ptr<ExecutionEngine>
create_libtriton_jit_engine(const EngineBuildContext& context,
                            AscendArtifact artifact);

}  // namespace flagdnn::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_ENGINES_LIBTRITON_JIT_HPP_
