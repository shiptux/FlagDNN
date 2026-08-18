/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/python_stdout_containment.hpp"

#include <Python.h>

#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

[[nodiscard]] std::string consume_python_error() {
  if (PyErr_Occurred() == nullptr) {
    return "unknown Python error";
  }
  PyObject* type = nullptr;
  PyObject* value = nullptr;
  PyObject* traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);
  PyObject* source = value != nullptr ? value : type;
  PyObject* rendered = source == nullptr ? nullptr : PyObject_Str(source);
  std::string result = "unprintable Python error";
  if (rendered != nullptr) {
    const char* text = PyUnicode_AsUTF8(rendered);
    if (text != nullptr) {
      result = text;
    }
  }
  Py_XDECREF(rendered);
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  PyErr_Clear();
  return result;
}

void run_python(const char* source) {
  if (PyRun_SimpleString(source) != 0) {
    throw std::runtime_error("embedded Python snippet failed: " +
                             consume_python_error());
  }
}

[[nodiscard]] PyObject* make_string_io() {
  PyObject* io_module = PyImport_ImportModule("io");
  if (io_module == nullptr) {
    throw std::runtime_error("cannot import io: " + consume_python_error());
  }
  PyObject* factory = PyObject_GetAttrString(io_module, "StringIO");
  Py_DECREF(io_module);
  if (factory == nullptr) {
    throw std::runtime_error("cannot find io.StringIO: " +
                             consume_python_error());
  }
  PyObject* result = PyObject_CallNoArgs(factory);
  Py_DECREF(factory);
  if (result == nullptr) {
    throw std::runtime_error("cannot create io.StringIO: " +
                             consume_python_error());
  }
  return result;
}

[[nodiscard]] std::string string_io_contents(PyObject* stream) {
  PyObject* value = PyObject_CallMethod(stream, "getvalue", nullptr);
  if (value == nullptr) {
    throw std::runtime_error("StringIO.getvalue failed: " +
                             consume_python_error());
  }
  const char* text = PyUnicode_AsUTF8(value);
  if (text == nullptr) {
    Py_DECREF(value);
    throw std::runtime_error("StringIO value is not UTF-8: " +
                             consume_python_error());
  }
  std::string result(text);
  Py_DECREF(value);
  return result;
}

void compiler_output(void*) {
  run_python(
      "payload = '[NPU] Generated arg_layout that must remain private'\n"
      "assert __import__('sys').stdout.write(payload) == len(payload)\n"
      "__import__('sys').stdout.flush()\n"
      "print('more private compiler output')\n"
      "print('compiler stderr remains visible', file=__import__('sys').stderr)\n");
  std::cout << "native stdout remains visible\n";
}

void failing_compiler_output(void*) {
  run_python("print('private output before callback failure')\n");
  throw std::runtime_error("callback failure sentinel");
}

[[nodiscard]] int fail(const std::string& message) {
  std::cerr << message << '\n';
  return 1;
}

void run_containment_from_released_main_thread(
    flagdnn::ascend::detail::ContainedPythonStdoutOperation operation) {
  PyThreadState* const main_state = PyEval_SaveThread();
  if (main_state == nullptr) {
    throw std::runtime_error("PyEval_SaveThread returned null");
  }

  std::exception_ptr failure;
  try {
    flagdnn::ascend::detail::run_with_contained_python_stdout(
        operation, nullptr);
  } catch (...) {
    failure = std::current_exception();
  }
  PyEval_RestoreThread(main_state);
  if (failure != nullptr) {
    std::rethrow_exception(failure);
  }
}

}  // namespace

int main() {
  Py_InitializeEx(0);
  if (Py_IsInitialized() == 0) {
    return fail("Py_InitializeEx failed");
  }

  PyObject* original_stdout = PySys_GetObject("stdout");
  PyObject* original_stderr = PySys_GetObject("stderr");
  if (original_stdout == nullptr || original_stderr == nullptr) {
    return fail("embedded Python has no standard streams");
  }
  Py_INCREF(original_stdout);
  Py_INCREF(original_stderr);

  PyObject* caller_stdout = nullptr;
  PyObject* caller_stderr = nullptr;
  int status = 0;
  try {
    caller_stdout = make_string_io();
    caller_stderr = make_string_io();
    if (PySys_SetObject("stdout", caller_stdout) != 0 ||
        PySys_SetObject("stderr", caller_stderr) != 0) {
      throw std::runtime_error("cannot install caller stream sentinels: " +
                               consume_python_error());
    }

    run_python("print('caller before')\n");
    std::ostringstream native_stdout;
    std::streambuf* const previous_buffer =
        std::cout.rdbuf(native_stdout.rdbuf());
    try {
      run_containment_from_released_main_thread(compiler_output);
    } catch (...) {
      std::cout.rdbuf(previous_buffer);
      throw;
    }
    std::cout.rdbuf(previous_buffer);

    if (PySys_GetObject("stdout") != caller_stdout) {
      throw std::runtime_error(
          "normal containment did not restore sys.stdout identity");
    }
    run_python("print('caller after')\n");
    if (string_io_contents(caller_stdout) !=
        "caller before\ncaller after\n") {
      throw std::runtime_error(
          "contained compiler output leaked into caller sys.stdout");
    }
    if (string_io_contents(caller_stderr) !=
        "compiler stderr remains visible\n") {
      throw std::runtime_error("Python stderr was unexpectedly contained");
    }
    if (native_stdout.str() != "native stdout remains visible\n") {
      throw std::runtime_error("native stdout was unexpectedly contained");
    }

    bool saw_original_failure = false;
    try {
      run_containment_from_released_main_thread(failing_compiler_output);
    } catch (const std::runtime_error& error) {
      saw_original_failure =
          std::string(error.what()) == "callback failure sentinel";
    }
    if (!saw_original_failure) {
      throw std::runtime_error(
          "callback exception semantics changed during restoration");
    }
    if (PySys_GetObject("stdout") != caller_stdout) {
      throw std::runtime_error(
          "exception containment did not restore sys.stdout identity");
    }
    run_python("print('ordinary later print')\n");
    if (string_io_contents(caller_stdout) !=
        "caller before\ncaller after\nordinary later print\n") {
      throw std::runtime_error(
          "stdout containment leaked into a later ordinary Python print");
    }
  } catch (const std::exception& error) {
    status = fail(error.what());
  }

  if (PySys_SetObject("stdout", original_stdout) != 0 ||
      PySys_SetObject("stderr", original_stderr) != 0) {
    status = fail("could not restore original embedded Python streams");
    PyErr_Clear();
  }
  Py_XDECREF(caller_stdout);
  Py_XDECREF(caller_stderr);
  Py_DECREF(original_stdout);
  Py_DECREF(original_stderr);
  if (Py_FinalizeEx() < 0) {
    status = fail("Py_FinalizeEx failed");
  }
  if (status == 0) {
    std::cout << "PASS Ascend embedded compiler stdout containment\n";
  }
  return status;
}
