/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/embedded_python.hpp"

#include <Python.h>

#include <stdexcept>
#include <string>

namespace flagdnn::ascend::detail {
namespace {

[[nodiscard]] std::string status_message(const PyStatus& status) {
  std::string result = status.err_msg == nullptr ? "unknown Python error"
                                                  : status.err_msg;
  if (status.func != nullptr) {
    result = std::string(status.func) + ": " + result;
  }
  return result;
}

}  // namespace

void initialize_embedded_python_from_program(const char* program_name) {
  if (program_name == nullptr || program_name[0] == '\0' ||
      Py_IsInitialized() != 0) {
    throw std::invalid_argument(
        "embedded Python requires a program and a clean runtime");
  }

  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  config.install_signal_handlers = 0;
  PyStatus status =
      PyConfig_SetBytesString(&config, &config.program_name, program_name);
  if (PyStatus_Exception(status)) {
    const std::string message = status_message(status);
    PyConfig_Clear(&config);
    throw std::runtime_error(
        "cannot configure embedded Python program: " + message);
  }
  status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status) || Py_IsInitialized() == 0) {
    throw std::runtime_error(
        "cannot initialize embedded Python: " + status_message(status));
  }
}

}  // namespace flagdnn::ascend::detail
