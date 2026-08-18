/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ENGINES_PYTHON_STDOUT_CONTAINMENT_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ENGINES_PYTHON_STDOUT_CONTAINMENT_HPP_

namespace flagdnn::ascend::detail {

using ContainedPythonStdoutOperation = void (*)(void* context);

/*
 * Run one embedded-Python compiler operation with Python-level stdout
 * redirected to a private /dev/null text stream.  The caller must hold the
 * Ascend process LTJ mutex.  The controlling thread holds the GIL from the
 * sys.stdout replacement through restoration; an extension called by the
 * callback may explicitly release it, which is safe only because Ascend owns a
 * clean-parent embedded-Python domain serialized by that mutex.  A callback
 * exception is rethrown unchanged when restoration succeeds.
 *
 * This deliberately affects only Python's sys.stdout.  Native stdout and
 * Python/native stderr remain caller-visible.
 */
void run_with_contained_python_stdout(
    ContainedPythonStdoutOperation operation,
    void* context);

}  // namespace flagdnn::ascend::detail

#endif  // FLAGDNN_BACKENDS_ASCEND_ENGINES_PYTHON_STDOUT_CONTAINMENT_HPP_
