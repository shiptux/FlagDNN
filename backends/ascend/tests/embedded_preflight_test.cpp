/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <Python.h>

#include <triton_jit/triton_jit_function.h>
#include <triton_jit/triton_kernel.h>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

bool set_environment(const char* name, const std::string& value) {
  if (::setenv(name, value.c_str(), 1) == 0) {
    return true;
  }
  std::cerr << "cannot set " << name << '\n';
  return false;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr << "usage: embedded_preflight cache ascend_binary "
                 "standalone_compile workspace_attestation module_root\n";
    return 2;
  }
  if (Py_IsInitialized() != 0) {
    std::cerr << "preflight parent is not Python-clean\n";
    return 3;
  }

  const std::filesystem::path cache = argv[1];
  std::error_code error;
  std::filesystem::remove_all(cache, error);
  error.clear();
  if (!std::filesystem::create_directories(cache / "tmp", error) || error) {
    std::cerr << "cannot create isolated preflight cache\n";
    return 4;
  }
  if (!set_environment("TRITON_JIT_BACKEND", "NPU") ||
      !set_environment("TRITON_BACKEND", "torch_npu") ||
      !set_environment("TRITON_ALL_BLOCKS_PARALLEL", "false") ||
      !set_environment("TORCH_DEVICE_BACKEND_AUTOLOAD", "0") ||
      !set_environment("PYTHONDONTWRITEBYTECODE", "1") ||
      !set_environment("PYTHONNOUSERSITE", "1") ||
      !set_environment("PYTHONHASHSEED", "0") ||
      !set_environment("PYTHONPATH", argv[5]) ||
      !set_environment("TRITON_CACHE_DIR", cache.string()) ||
      !set_environment("TMPDIR", (cache / "tmp").string()) ||
      !set_environment("FLAGDNN_PREFLIGHT_STANDALONE", argv[3]) ||
      !set_environment("FLAGDNN_PREFLIGHT_ATTESTATION", argv[4])) {
    return 5;
  }

  try {
    triton_jit::set_launch_enter_hook(
        [](const triton_jit::LaunchMetadata&) {});
    triton_jit::clear_launch_hooks();

    triton_jit::TritonJITFunction& function =
        triton_jit::TritonJITFunction::get_instance(
            argv[2], "binary_contiguous_kernel");
    if (function.get_static_sig().num_args != 8 || Py_IsInitialized() == 0) {
      std::cerr << "libtriton_jit/Ascend binary signature preflight failed\n";
      return 6;
    }

    constexpr const char* kImportProbe = R"PY(
import importlib.util
import os
import torch
import triton
import yaml
import torch_npu

def load_exact(name, environment):
    path = os.environ[environment]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

standalone = load_exact("_flagdnn_preflight_standalone", "FLAGDNN_PREFLIGHT_STANDALONE")
attestation = load_exact("_flagdnn_preflight_attestation", "FLAGDNN_PREFLIGHT_ATTESTATION")
if not callable(getattr(standalone, "compile_a_kernel", None)):
    raise RuntimeError("standalone compiler has no compile_a_kernel")
if not callable(getattr(attestation, "main", None)):
    raise RuntimeError("workspace attestation has no main")
)PY";
    const PyGILState_STATE gil = PyGILState_Ensure();
    const int import_result = PyRun_SimpleString(kImportProbe);
    PyGILState_Release(gil);
    if (import_result != 0) {
      std::cerr << "embedded Python dependency import failed\n";
      return 7;
    }
  } catch (const std::exception& exception) {
    std::cerr << "embedded preflight exception: " << exception.what() << '\n';
    return 8;
  } catch (...) {
    std::cerr << "unknown embedded preflight exception\n";
    return 9;
  }
  return 0;
}
