/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/hygon/engines/engine.hpp"

#include "backends/hygon/error.hpp"

#include <utility>

namespace flagdnn::hygon {

std::unique_ptr<ExecutionEngine>
create_execution_engine(const EngineBuildContext &context,
                        const flagdnnBackendBuildInputV2 &input) {
  HygonArtifact artifact = parse_hygon_artifact(context, input);
  switch (artifact.engine) {
  case EngineKind::kExternalArtifact:
    throw HygonError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "Hygon backend supports only libtriton_jit execution artifacts");
  case EngineKind::kLibTritonJit:
    return create_libtriton_jit_engine(context, std::move(artifact));
  }
  throw HygonError(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
                   "unknown HIP execution engine");
}

} // namespace flagdnn::hygon
