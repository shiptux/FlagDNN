/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/engine.hpp"

#include "backends/ascend/artifact.hpp"
#include "backends/ascend/engines/libtriton_jit.hpp"

namespace flagdnn::ascend {

std::unique_ptr<ExecutionEngine> create_execution_engine(
    const EngineBuildContext& context,
    const flagdnnBackendBuildInputV2& input) {
  return create_libtriton_jit_engine(
      context,
      parse_ascend_artifact(
          context.target_fingerprint, context.ai_core_count, input));
}

}  // namespace flagdnn::ascend
