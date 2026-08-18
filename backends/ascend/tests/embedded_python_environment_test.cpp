/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/embedded_python.hpp"

#include <Python.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;

fs::path embedded_purelib() {
  PyObject* sysconfig = PyImport_ImportModule("sysconfig");
  if (sysconfig == nullptr) {
    throw std::runtime_error("cannot import embedded sysconfig");
  }
  PyObject* get_paths = PyObject_GetAttrString(sysconfig, "get_paths");
  Py_DECREF(sysconfig);
  if (get_paths == nullptr || !PyCallable_Check(get_paths)) {
    Py_XDECREF(get_paths);
    throw std::runtime_error("embedded sysconfig.get_paths is unavailable");
  }
  PyObject* paths = PyObject_CallNoArgs(get_paths);
  Py_DECREF(get_paths);
  if (paths == nullptr || !PyDict_Check(paths)) {
    Py_XDECREF(paths);
    throw std::runtime_error("embedded sysconfig.get_paths failed");
  }
  PyObject* purelib = PyDict_GetItemString(paths, "purelib");
  const char* value =
      purelib == nullptr || !PyUnicode_Check(purelib)
          ? nullptr
          : PyUnicode_AsUTF8(purelib);
  if (value == nullptr) {
    Py_DECREF(paths);
    throw std::runtime_error("embedded sysconfig purelib is invalid");
  }
  const fs::path result(value);
  Py_DECREF(paths);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 || Py_IsInitialized() != 0) {
      throw std::runtime_error(
          "expected Python program, module root, and a clean runtime");
    }
    flagdnn::ascend::detail::initialize_embedded_python_from_program(argv[1]);
    if (Py_IsInitialized() == 0) {
      throw std::runtime_error("embedded Python did not initialize");
    }
    std::error_code error;
    const fs::path actual = fs::canonical(embedded_purelib(), error);
    if (error) {
      throw std::runtime_error("cannot canonicalize embedded purelib");
    }
    const fs::path expected = fs::canonical(argv[2], error);
    if (error || actual != expected) {
      throw std::runtime_error(
          "embedded Python did not select the configured venv purelib");
    }
    if (Py_FinalizeEx() < 0) {
      throw std::runtime_error("embedded Python finalization failed");
    }
    std::cout << "embedded Python venv selection passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
