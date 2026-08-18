/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ENGINES_BUILD_TIME_PREWARM_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ENGINES_BUILD_TIME_PREWARM_HPP_

#include <utility>

namespace flagdnn::ascend::detail {

template <typename Resources, typename Stage, typename Launch>
[[nodiscard]] auto run_checked_build_time_prewarm(Resources& resources,
                                                  const Stage& stage,
                                                  Launch&& launch)
    -> decltype(resources.read_stage_output(stage)) {
  resources.restore_stage_inputs(stage);
  resources.reset_stage_output(stage);
  std::forward<Launch>(launch)();
  auto output = resources.read_stage_output(stage);
  resources.validate_stage_inputs_unchanged(stage);
  resources.validate_stage_output(stage, output);
  return output;
}

}  // namespace flagdnn::ascend::detail

#endif  // FLAGDNN_BACKENDS_ASCEND_ENGINES_BUILD_TIME_PREWARM_HPP_
