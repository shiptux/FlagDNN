/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_ARTIFACT_HPP_
#define FLAGDNN_BACKENDS_HYGON_ARTIFACT_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "backends/backend_api.h"
#include "backends/hygon/context.hpp"

namespace flagdnn::hygon {

enum class EngineKind {
  kExternalArtifact,
  kLibTritonJit,
};

enum class ArgumentKind {
  kTensor,
  kWorkspaceTensor,
  kScalarI32,
  kScalarF32,
};

struct ArgumentSpec {
  ArgumentKind kind = ArgumentKind::kTensor;
  std::int64_t uid = 0;
  std::int32_t scalar_i32 = 0;
  float scalar_f32 = 0.0F;
  std::size_t workspace_offset = 0;
  std::size_t storage_size = 0;
  std::size_t alignment = 1;
  std::string semantic_name;
};

struct HygonKernelArtifact {
  std::string variant_id = "default";
  std::filesystem::path binary;
  std::string entry_symbol;
  std::string full_signature;
  unsigned int num_warps = 0;
  unsigned int num_stages = 0;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  std::array<unsigned int, 3> block = {1, 1, 1};
  unsigned int shared_memory = 0;
  std::size_t global_scratch_size = 0;
  std::size_t profile_scratch_size = 0;
  std::vector<ArgumentSpec> arguments;
  std::vector<std::int64_t> binding_uids;
};

struct HygonStageArtifact {
  std::filesystem::path source;
  std::string materialized_source;
  std::string materialized_source_sha256;
  std::string function_name;
  std::vector<HygonKernelArtifact> variants;
  bool autotune = false;
  unsigned int warmup = 0;
  unsigned int repetitions = 1;
  std::string candidate_identity;
  std::filesystem::path selection_cache;
};

struct HygonArtifact {
  EngineKind engine = EngineKind::kExternalArtifact;
  std::vector<HygonStageArtifact> stages;
  std::vector<std::int64_t> binding_uids;
  std::size_t workspace_size = 0;
  std::size_t workspace_alignment = 1;
};

[[nodiscard]] HygonArtifact
parse_hygon_artifact(const EngineBuildContext &context,
                     const flagdnnBackendBuildInputV2 &input);

} // namespace flagdnn::hygon

#endif // FLAGDNN_BACKENDS_HYGON_ARTIFACT_HPP_
