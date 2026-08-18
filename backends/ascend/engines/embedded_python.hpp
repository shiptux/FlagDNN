/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ENGINES_EMBEDDED_PYTHON_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ENGINES_EMBEDDED_PYTHON_HPP_

namespace flagdnn::ascend::detail {

void initialize_embedded_python_from_program(const char* program_name);

}  // namespace flagdnn::ascend::detail

#endif  // FLAGDNN_BACKENDS_ASCEND_ENGINES_EMBEDDED_PYTHON_HPP_
