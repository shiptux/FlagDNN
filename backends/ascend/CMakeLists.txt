# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

if(NOT FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
  message(FATAL_ERROR
    "FlagDNN Ascend supports only FLAGDNN_EXECUTION_ENGINE=libtriton_jit")
endif()

set(CANN_ROOT "" CACHE PATH "Root of the CANN 9 installation")
if(NOT CANN_ROOT)
  foreach(environment_name IN ITEMS ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME)
    if(DEFINED ENV{${environment_name}} AND
       NOT "$ENV{${environment_name}}" STREQUAL "")
      set(CANN_ROOT "$ENV{${environment_name}}" CACHE PATH
        "Root of the CANN 9 installation" FORCE)
      break()
    endif()
  endforeach()
endif()
if(NOT CANN_ROOT)
  message(FATAL_ERROR
    "Ascend backend requires CANN_ROOT, ASCEND_HOME_PATH, or "
    "ASCEND_TOOLKIT_HOME")
endif()

include(cmake/TritonJITCompat.cmake)
flagdnn_ascend_prepare_triton_jit("${CANN_ROOT}")

if(NOT FLAGDNN_CODEGEN_PYTHON)
  message(FATAL_ERROR
    "Ascend libtriton_jit requires FLAGDNN_CODEGEN_PYTHON")
endif()
set(Python_EXECUTABLE "${FLAGDNN_CODEGEN_PYTHON}")
find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -I -c
    "import importlib.util,sys; names=('torch','triton','yaml','torch_npu'); missing=[n for n in names if importlib.util.find_spec(n) is None]; print(','.join(missing)); sys.exit(bool(missing))"
  TIMEOUT 10
  RESULT_VARIABLE _flagdnn_ascend_module_result
  OUTPUT_VARIABLE _flagdnn_ascend_missing_modules
  ERROR_VARIABLE _flagdnn_ascend_module_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _flagdnn_ascend_module_result EQUAL 0)
  message(FATAL_ERROR
    "FLAGDNN_CODEGEN_PYTHON is missing required Ascend compiler modules "
    "(${_flagdnn_ascend_missing_modules}): ${_flagdnn_ascend_module_error}")
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" -I -c
    "import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()['purelib']).resolve())"
  TIMEOUT 10
  RESULT_VARIABLE _flagdnn_ascend_python_module_root_result
  OUTPUT_VARIABLE _flagdnn_ascend_python_module_root
  ERROR_VARIABLE _flagdnn_ascend_python_module_root_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _flagdnn_ascend_python_module_root_result EQUAL 0 OR
   NOT IS_DIRECTORY "${_flagdnn_ascend_python_module_root}")
  message(FATAL_ERROR
    "Cannot resolve the selected Ascend Python module root: "
    "${_flagdnn_ascend_python_module_root_error}")
endif()
file(REAL_PATH "${_flagdnn_ascend_python_module_root}"
  _flagdnn_ascend_python_module_root)
file(REAL_PATH "${Python_EXECUTABLE}"
  _flagdnn_ascend_python_program_canonical)
if(NOT EXISTS "${_flagdnn_ascend_python_program_canonical}")
  message(FATAL_ERROR "Cannot resolve the selected Ascend Python program")
endif()
file(SHA256 "${_flagdnn_ascend_python_program_canonical}"
  _flagdnn_ascend_python_program_sha256)

# A venv may intentionally use system-site-packages: the NPU Triton package
# can live in the venv while torch/torch_npu/PyYAML live in its pinned base
# interpreter.  Record every selected module origin independently rather than
# weakening the runtime check to accept any importable module on sys.path.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -I -c
    "import importlib.util,pathlib; names=('torch','triton','torch_npu','yaml'); print('\\n'.join(str(pathlib.Path(importlib.util.find_spec(n).origin).resolve()) for n in names))"
  TIMEOUT 10
  RESULT_VARIABLE _flagdnn_ascend_module_origins_result
  OUTPUT_VARIABLE _flagdnn_ascend_module_origins_output
  ERROR_VARIABLE _flagdnn_ascend_module_origins_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" _flagdnn_ascend_module_origins
  "${_flagdnn_ascend_module_origins_output}")
list(LENGTH _flagdnn_ascend_module_origins
  _flagdnn_ascend_module_origin_count)
if(NOT _flagdnn_ascend_module_origins_result EQUAL 0 OR
   NOT _flagdnn_ascend_module_origin_count EQUAL 4)
  message(FATAL_ERROR
    "Cannot resolve exact Ascend Python module origins: "
    "${_flagdnn_ascend_module_origins_error}")
endif()
list(GET _flagdnn_ascend_module_origins 0 _flagdnn_ascend_torch_module)
list(GET _flagdnn_ascend_module_origins 1 _flagdnn_ascend_triton_module)
list(GET _flagdnn_ascend_module_origins 2 _flagdnn_ascend_torch_npu_module)
list(GET _flagdnn_ascend_module_origins 3 _flagdnn_ascend_yaml_module)
foreach(module_path IN ITEMS
    "${_flagdnn_ascend_torch_module}"
    "${_flagdnn_ascend_triton_module}"
    "${_flagdnn_ascend_torch_npu_module}"
    "${_flagdnn_ascend_yaml_module}")
  if(NOT IS_ABSOLUTE "${module_path}" OR NOT EXISTS "${module_path}")
    message(FATAL_ERROR
      "Selected Ascend Python module origin is not a regular absolute file: "
      "${module_path}")
  endif()
endforeach()
file(SHA256 "${_flagdnn_ascend_torch_module}"
  _flagdnn_ascend_torch_module_sha256)
file(SHA256 "${_flagdnn_ascend_triton_module}"
  _flagdnn_ascend_triton_module_sha256)
file(SHA256 "${_flagdnn_ascend_torch_npu_module}"
  _flagdnn_ascend_torch_npu_module_sha256)
file(SHA256 "${_flagdnn_ascend_yaml_module}"
  _flagdnn_ascend_yaml_module_sha256)

if(NOT DEFINED TritonJIT_DIR OR TritonJIT_DIR STREQUAL "")
  set(_flagdnn_ascend_triton_jit_candidates)
  if(DEFINED ENV{TritonJIT_DIR} AND NOT "$ENV{TritonJIT_DIR}" STREQUAL "")
    list(APPEND _flagdnn_ascend_triton_jit_candidates "$ENV{TritonJIT_DIR}")
  endif()
  if(DEFINED ENV{LIBTRITON_JIT_ROOT} AND
     NOT "$ENV{LIBTRITON_JIT_ROOT}" STREQUAL "")
    list(APPEND _flagdnn_ascend_triton_jit_candidates
      "$ENV{LIBTRITON_JIT_ROOT}/build"
      "$ENV{LIBTRITON_JIT_ROOT}/lib/cmake/TritonJIT")
  endif()
  list(APPEND _flagdnn_ascend_triton_jit_candidates
    "${PROJECT_SOURCE_DIR}/../libtriton_jit/build")
  foreach(candidate IN LISTS _flagdnn_ascend_triton_jit_candidates)
    if(EXISTS "${candidate}/TritonJITConfig.cmake")
      file(REAL_PATH "${candidate}" candidate)
      set(TritonJIT_DIR "${candidate}" CACHE PATH
        "Directory containing NPU TritonJITConfig.cmake")
      break()
    endif()
  endforeach()
endif()
if(NOT DEFINED TritonJIT_DIR OR TritonJIT_DIR STREQUAL "")
  message(FATAL_ERROR "Cannot find the installed NPU TritonJIT package")
endif()

foreach(module_candidate IN ITEMS
    "${TritonJIT_DIR}"
    "${TritonJIT_DIR}/cmake"
    "${TritonJIT_DIR}/../cmake")
  if(EXISTS "${module_candidate}/FindTorch.cmake")
    file(REAL_PATH "${module_candidate}" module_candidate)
    list(PREPEND CMAKE_MODULE_PATH "${module_candidate}")
    break()
  endif()
endforeach()

# TritonJIT's FindTorch module treats torch.utils.cmake_prefix_path as
# Torch_ROOT, although current Torch packages place TorchConfig.cmake in its
# Torch/ child. Locate without importing torch (which may initialize the NPU
# runtime), then reject a stale/user-supplied package from another interpreter.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -I -c
    "import importlib.util,pathlib; s=importlib.util.find_spec('torch'); print(pathlib.Path(s.origin).parent/'share'/'cmake'/'Torch' if s and s.origin else '')"
  TIMEOUT 10
  RESULT_VARIABLE _flagdnn_ascend_torch_result
  OUTPUT_VARIABLE _flagdnn_ascend_torch_output
  ERROR_VARIABLE _flagdnn_ascend_torch_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" _flagdnn_ascend_torch_lines
  "${_flagdnn_ascend_torch_output}")
set(_flagdnn_ascend_torch_directory "")
foreach(candidate IN LISTS _flagdnn_ascend_torch_lines)
  string(STRIP "${candidate}" candidate)
  if(EXISTS "${candidate}/TorchConfig.cmake")
    file(REAL_PATH "${candidate}" _flagdnn_ascend_torch_directory)
  endif()
endforeach()
if(NOT _flagdnn_ascend_torch_result EQUAL 0 OR
   NOT _flagdnn_ascend_torch_directory)
  message(FATAL_ERROR
    "FLAGDNN_CODEGEN_PYTHON does not provide a usable Torch CMake package; "
    "reported '${_flagdnn_ascend_torch_output}': "
    "${_flagdnn_ascend_torch_error}")
endif()
if(Torch_DIR AND EXISTS "${Torch_DIR}")
  file(REAL_PATH "${Torch_DIR}" _flagdnn_ascend_requested_torch_directory)
  if(NOT _flagdnn_ascend_requested_torch_directory STREQUAL
         _flagdnn_ascend_torch_directory)
    message(FATAL_ERROR
      "Torch_DIR belongs to a different Python environment: "
      "${_flagdnn_ascend_requested_torch_directory}")
  endif()
endif()
set(Torch_DIR "${_flagdnn_ascend_torch_directory}" CACHE PATH
  "Torch CMake package used by NPU libtriton_jit" FORCE)
get_filename_component(_flagdnn_ascend_torch_cmake_root "${Torch_DIR}"
  DIRECTORY)
set(Torch_ROOT "${_flagdnn_ascend_torch_cmake_root}" CACHE PATH
  "Torch CMake prefix used by NPU libtriton_jit" FORCE)

find_package(TritonJIT 0.1.0 CONFIG REQUIRED)
if(NOT TritonJIT_BACKEND STREQUAL "NPU")
  message(FATAL_ERROR
    "FlagDNN Ascend requires TritonJIT_BACKEND=NPU; found "
    "${TritonJIT_BACKEND}")
endif()
if(TARGET TritonJIT::triton_jit)
  set(_flagdnn_ascend_triton_jit_target TritonJIT::triton_jit)
elseif(TARGET triton_jit)
  set(_flagdnn_ascend_triton_jit_target triton_jit)
else()
  message(FATAL_ERROR
    "TritonJIT was found but exported no usable triton_jit target")
endif()
get_target_property(_flagdnn_ascend_triton_jit_imported
  "${_flagdnn_ascend_triton_jit_target}" IMPORTED)
if(_flagdnn_ascend_triton_jit_imported)
  # Validation is assembled later from a sibling directory and must inspect
  # the exact production LTJ identity without re-running package discovery.
  set_property(TARGET "${_flagdnn_ascend_triton_jit_target}"
    PROPERTY IMPORTED_GLOBAL TRUE)
endif()

function(_flagdnn_ascend_imported_location target output)
  set(location "")
  foreach(property IN ITEMS
      IMPORTED_LOCATION IMPORTED_LOCATION_NOCONFIG IMPORTED_LOCATION_RELEASE
      IMPORTED_LOCATION_RELWITHDEBINFO IMPORTED_LOCATION_DEBUG)
    if(NOT location)
      get_target_property(candidate "${target}" "${property}")
      if(candidate AND EXISTS "${candidate}")
        file(REAL_PATH "${candidate}" location)
      endif()
    endif()
  endforeach()
  if(NOT location)
    message(FATAL_ERROR "Cannot determine imported location for ${target}")
  endif()
  set("${output}" "${location}" PARENT_SCOPE)
endfunction()

_flagdnn_ascend_imported_location(
  "${_flagdnn_ascend_triton_jit_target}" _flagdnn_ascend_triton_jit_library)
_flagdnn_ascend_imported_location(Python::Python _flagdnn_ascend_python_library)
get_target_property(_flagdnn_ascend_ltj_include_candidates
  "${_flagdnn_ascend_triton_jit_target}" INTERFACE_INCLUDE_DIRECTORIES)
set(_flagdnn_ascend_ltj_include "")
foreach(candidate IN LISTS _flagdnn_ascend_ltj_include_candidates)
  if(IS_DIRECTORY "${candidate}" AND
     EXISTS "${candidate}/triton_jit/kernel_metadata.h")
    file(REAL_PATH "${candidate}" candidate)
    if(_flagdnn_ascend_ltj_include AND
       NOT _flagdnn_ascend_ltj_include STREQUAL "${candidate}")
      message(FATAL_ERROR
        "TritonJIT exports multiple incompatible public include roots")
    endif()
    set(_flagdnn_ascend_ltj_include "${candidate}")
  endif()
endforeach()
if(NOT _flagdnn_ascend_ltj_include)
  message(FATAL_ERROR
    "Cannot locate the public headers of the selected NPU TritonJIT")
endif()
set(FLAGDNN_TRITON_JIT_STANDALONE_COMPILER "" CACHE FILEPATH
  "Exact standalone_compile.py paired with NPU TritonJIT")
if(NOT FLAGDNN_TRITON_JIT_STANDALONE_COMPILER)
  get_filename_component(_flagdnn_ascend_ltj_root
    "${_flagdnn_ascend_ltj_include}" DIRECTORY)
  set(_flagdnn_ascend_standalone_candidate
    "${_flagdnn_ascend_ltj_root}/scripts/standalone_compile.py")
  if(EXISTS "${_flagdnn_ascend_standalone_candidate}")
    set(FLAGDNN_TRITON_JIT_STANDALONE_COMPILER
      "${_flagdnn_ascend_standalone_candidate}" CACHE FILEPATH
      "Exact standalone_compile.py paired with NPU TritonJIT" FORCE)
  endif()
endif()
if(NOT EXISTS "${FLAGDNN_TRITON_JIT_STANDALONE_COMPILER}" OR
   NOT EXISTS "${TritonJIT_DIR}/TritonJITConfig.cmake")
  message(FATAL_ERROR
    "NPU TritonJIT standalone compiler/config identity is incomplete")
endif()
file(REAL_PATH "${FLAGDNN_TRITON_JIT_STANDALONE_COMPILER}"
  _flagdnn_ascend_ltj_standalone)
file(REAL_PATH "${TritonJIT_DIR}/TritonJITConfig.cmake"
  _flagdnn_ascend_ltj_config)
file(REAL_PATH "${_flagdnn_ascend_python_library}"
  _flagdnn_ascend_python_library)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -I -c
    "import os,sysconfig; print(os.path.realpath(os.path.join(sysconfig.get_config_var('LIBDIR'),sysconfig.get_config_var('LDLIBRARY'))))"
  TIMEOUT 10
  RESULT_VARIABLE _flagdnn_ascend_python_library_result
  OUTPUT_VARIABLE _flagdnn_ascend_interpreter_library
  ERROR_VARIABLE _flagdnn_ascend_python_library_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _flagdnn_ascend_python_library_result EQUAL 0 OR
   NOT EXISTS "${_flagdnn_ascend_interpreter_library}")
  message(FATAL_ERROR
    "Cannot resolve libpython for FLAGDNN_CODEGEN_PYTHON: "
    "${_flagdnn_ascend_python_library_error}")
endif()
file(REAL_PATH "${_flagdnn_ascend_interpreter_library}"
  _flagdnn_ascend_interpreter_library)
if(NOT _flagdnn_ascend_interpreter_library STREQUAL
       _flagdnn_ascend_python_library)
  message(FATAL_ERROR
    "Python::Python (${_flagdnn_ascend_python_library}) differs from the "
    "runtime library of FLAGDNN_CODEGEN_PYTHON "
    "(${_flagdnn_ascend_interpreter_library})")
endif()
get_filename_component(_flagdnn_ascend_python_soname
  "${_flagdnn_ascend_python_library}" NAME)
if(NOT CMAKE_READELF)
  find_program(CMAKE_READELF readelf REQUIRED)
endif()
execute_process(
  COMMAND "${CMAKE_READELF}" -d "${_flagdnn_ascend_triton_jit_library}"
  RESULT_VARIABLE _flagdnn_ascend_readelf_result
  OUTPUT_VARIABLE _flagdnn_ascend_readelf_output
  ERROR_VARIABLE _flagdnn_ascend_readelf_error)
if(NOT _flagdnn_ascend_readelf_result EQUAL 0)
  message(FATAL_ERROR
    "Cannot inspect libtriton_jit dependencies: "
    "${_flagdnn_ascend_readelf_error}")
endif()
string(FIND "${_flagdnn_ascend_readelf_output}"
  "[${_flagdnn_ascend_python_soname}]" _flagdnn_ascend_python_needed)
if(_flagdnn_ascend_python_needed EQUAL -1)
  message(FATAL_ERROR
    "Python::Python (${_flagdnn_ascend_python_soname}) does not match the "
    "libpython ABI required by NPU libtriton_jit")
endif()

file(SHA256 "${FLAGDNN_ASCEND_ASCENDCL_LIBRARY}"
  _flagdnn_ascend_ascendcl_sha256)
file(SHA256 "${FLAGDNN_ASCEND_RUNTIME_LIBRARY}"
  _flagdnn_ascend_runtime_sha256)
file(SHA256 "${_flagdnn_ascend_triton_jit_library}"
  _flagdnn_ascend_triton_jit_sha256)
file(SHA256 "${_flagdnn_ascend_python_library}"
  _flagdnn_ascend_python_library_sha256)
file(SHA256 "${_flagdnn_ascend_ltj_standalone}"
  _flagdnn_ascend_ltj_standalone_sha256)
set(_flagdnn_ascend_identity_definitions
  FLAGDNN_ASCEND_CANN_ROOT="${FLAGDNN_ASCEND_CANN_ROOT}"
  FLAGDNN_ASCEND_CANN_VERSION="${FLAGDNN_ASCEND_CANN_VERSION}"
  FLAGDNN_ASCEND_CANN_INNER_VERSION="${FLAGDNN_ASCEND_CANN_INNER_VERSION}"
  FLAGDNN_ASCEND_ASCENDCL_PATH="${FLAGDNN_ASCEND_ASCENDCL_LIBRARY}"
  FLAGDNN_ASCEND_ASCENDCL_SHA256="${_flagdnn_ascend_ascendcl_sha256}"
  FLAGDNN_ASCEND_RUNTIME_PATH="${FLAGDNN_ASCEND_RUNTIME_LIBRARY}"
  FLAGDNN_ASCEND_RUNTIME_SHA256="${_flagdnn_ascend_runtime_sha256}"
  FLAGDNN_ASCEND_LIBTRITON_JIT_PATH="${_flagdnn_ascend_triton_jit_library}"
  FLAGDNN_ASCEND_LIBTRITON_JIT_SHA256="${_flagdnn_ascend_triton_jit_sha256}"
  FLAGDNN_ASCEND_PYTHON_LIBRARY_PATH="${_flagdnn_ascend_python_library}"
  FLAGDNN_ASCEND_PYTHON_LIBRARY_SHA256="${_flagdnn_ascend_python_library_sha256}"
  FLAGDNN_ASCEND_STANDALONE_PATH="${_flagdnn_ascend_ltj_standalone}"
  FLAGDNN_ASCEND_STANDALONE_SHA256="${_flagdnn_ascend_ltj_standalone_sha256}"
  FLAGDNN_ASCEND_PYTHON_MODULE_ROOT="${_flagdnn_ascend_python_module_root}"
  FLAGDNN_ASCEND_PYTHON_PROGRAM_PATH="${Python_EXECUTABLE}"
  FLAGDNN_ASCEND_PYTHON_PROGRAM_CANONICAL_PATH="${_flagdnn_ascend_python_program_canonical}"
  FLAGDNN_ASCEND_PYTHON_PROGRAM_SHA256="${_flagdnn_ascend_python_program_sha256}"
  FLAGDNN_ASCEND_TORCH_MODULE_PATH="${_flagdnn_ascend_torch_module}"
  FLAGDNN_ASCEND_TORCH_MODULE_SHA256="${_flagdnn_ascend_torch_module_sha256}"
  FLAGDNN_ASCEND_TRITON_MODULE_PATH="${_flagdnn_ascend_triton_module}"
  FLAGDNN_ASCEND_TRITON_MODULE_SHA256="${_flagdnn_ascend_triton_module_sha256}"
  FLAGDNN_ASCEND_TORCH_NPU_MODULE_PATH="${_flagdnn_ascend_torch_npu_module}"
  FLAGDNN_ASCEND_TORCH_NPU_MODULE_SHA256="${_flagdnn_ascend_torch_npu_module_sha256}"
  FLAGDNN_ASCEND_YAML_MODULE_PATH="${_flagdnn_ascend_yaml_module}"
  FLAGDNN_ASCEND_YAML_MODULE_SHA256="${_flagdnn_ascend_yaml_module_sha256}")

# The core owns the external compiler process. Preserve the configure-time
# canonical toolkit root so a deployment configured with CANN_ROOT (rather
# than ambient shell variables) still gives the sanitized child the exact
# ASCEND_HOME_PATH used to build the plugin.
target_compile_definitions(flagdnn PRIVATE
  FLAGDNN_ASCEND_DEFAULT_CANN_ROOT="${FLAGDNN_ASCEND_CANN_ROOT}"
  FLAGDNN_ASCEND_DEFAULT_LIBTRITON_JIT_LIBRARY="${_flagdnn_ascend_triton_jit_library}"
  FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_STANDALONE_COMPILER="${_flagdnn_ascend_ltj_standalone}"
  FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_CONFIG="${_flagdnn_ascend_ltj_config}"
  FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_INCLUDE_DIRECTORY="${_flagdnn_ascend_ltj_include}"
  FLAGDNN_ASCEND_DEFAULT_PYTHON_MODULE_ROOT="${_flagdnn_ascend_python_module_root}")

# The public NPU TritonJIT headers include CANN's architecture-specific
# runtime/runtime/rt.h.  Keep this exact configure-time CANN include root on
# the production plugin as well as on the standalone embedded preflight.
find_path(_flagdnn_ascend_runtime_header_root runtime/runtime/rt.h
  PATHS
    "${FLAGDNN_ASCEND_CANN_ROOT}/aarch64-linux/pkg_inc"
    "${FLAGDNN_ASCEND_CANN_ROOT}/x86_64-linux/pkg_inc"
  NO_DEFAULT_PATH REQUIRED)
get_filename_component(_flagdnn_ascend_arch_root
  "${_flagdnn_ascend_runtime_header_root}" DIRECTORY)

flagdnn_add_backend_plugin(ascend INSTALL ABI_VERSION 3
  SOURCES
    artifact.cpp
    backend.cpp
    context.cpp
    engines/engine.cpp
    engines/embedded_python.cpp
    engines/libtriton_jit.cpp
    engines/batchnorm_oracle.cpp
    engines/batchnorm_inference_oracle.cpp
    engines/layernorm_oracle.cpp
    engines/matmul_oracle.cpp
    engines/convolution_fprop_oracle.cpp
    engines/rmsnorm_oracle.cpp
    engines/python_stdout_containment.cpp
    engines/reduction_oracle.cpp
    error.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/json.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp
  LINK_LIBRARIES
    Ascend::ascendcl
    Ascend::runtime
    ${_flagdnn_ascend_triton_jit_target}
    Python::Python
    ${CMAKE_DL_LIBS})
target_compile_definitions(flagdnn_backend_ascend PRIVATE
  ${_flagdnn_ascend_identity_definitions})
target_include_directories(flagdnn_backend_ascend SYSTEM PRIVATE
  "${FLAGDNN_ASCEND_CANN_ROOT}/include"
  "${FLAGDNN_ASCEND_CANN_ROOT}/include/aclnn"
  "${FLAGDNN_ASCEND_CANN_ROOT}/include/experiment"
  "${FLAGDNN_ASCEND_CANN_ROOT}/include/experiment/runtime"
  "${_flagdnn_ascend_arch_root}/include"
  "${_flagdnn_ascend_arch_root}/include/experiment"
  "${_flagdnn_ascend_arch_root}/include/experiment/msprof"
  "${_flagdnn_ascend_arch_root}/include/experiment/msprof/toolchain"
  "${_flagdnn_ascend_runtime_header_root}"
  "${_flagdnn_ascend_runtime_header_root}/runtime/runtime")
set_target_properties(flagdnn_backend_ascend PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  INSTALL_RPATH_USE_LINK_PATH NO)

install(FILES kernels/registry.json
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/ascend/kernels")
foreach(resource IN ITEMS add_plan.py compiler.py compiler_identity.py
                          capabilities.json sandbox_policy.json
                          tuning_decoder.py workspace_attestation.py)
  install(FILES "${resource}"
    DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/ascend")
endforeach()
install(DIRECTORY kernels/
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/ascend/kernels"
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY tuning/
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/ascend/tuning"
  FILES_MATCHING PATTERN "*.yaml")

if(FLAGDNN_BUILD_TESTS)
  find_package(Threads REQUIRED)
  add_executable(flagdnn_test_ascend_batchnorm_oracle
    tests/batchnorm_oracle_test.cpp
    engines/batchnorm_oracle.cpp)
  target_compile_features(flagdnn_test_ascend_batchnorm_oracle PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_batchnorm_oracle PRIVATE
    ${PROJECT_SOURCE_DIR})
  flagdnn_enable_warnings(flagdnn_test_ascend_batchnorm_oracle)
  add_test(NAME core.ascend.batchnorm_host_oracle
    COMMAND flagdnn_test_ascend_batchnorm_oracle)
  set_tests_properties(core.ascend.batchnorm_host_oracle PROPERTIES
    LABELS "core;ascend;no_device" TIMEOUT 30)

  add_executable(flagdnn_test_ascend_layernorm_oracle
    tests/layernorm_oracle_test.cpp
    engines/layernorm_oracle.cpp)
  target_compile_features(flagdnn_test_ascend_layernorm_oracle PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_layernorm_oracle PRIVATE
    ${PROJECT_SOURCE_DIR})
  flagdnn_enable_warnings(flagdnn_test_ascend_layernorm_oracle)
  add_test(NAME core.ascend.layernorm_host_oracle
    COMMAND flagdnn_test_ascend_layernorm_oracle)
  set_tests_properties(core.ascend.layernorm_host_oracle PROPERTIES
    LABELS "core;ascend;no_device" TIMEOUT 30)

  add_executable(flagdnn_test_ascend_rmsnorm_oracle
    tests/rmsnorm_oracle_test.cpp
    engines/rmsnorm_oracle.cpp)
  target_compile_features(flagdnn_test_ascend_rmsnorm_oracle PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_rmsnorm_oracle PRIVATE
    ${PROJECT_SOURCE_DIR})
  flagdnn_enable_warnings(flagdnn_test_ascend_rmsnorm_oracle)
  add_test(NAME core.ascend.rmsnorm_host_oracle
    COMMAND flagdnn_test_ascend_rmsnorm_oracle)
  set_tests_properties(core.ascend.rmsnorm_host_oracle PROPERTIES
    LABELS "core;ascend;no_device" TIMEOUT 30)

  add_executable(flagdnn_test_ascend_batchnorm_inference_oracle
    tests/batchnorm_inference_oracle_test.cpp
    engines/batchnorm_inference_oracle.cpp)
  target_compile_features(flagdnn_test_ascend_batchnorm_inference_oracle PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_batchnorm_inference_oracle PRIVATE
    ${PROJECT_SOURCE_DIR})
  flagdnn_enable_warnings(flagdnn_test_ascend_batchnorm_inference_oracle)
  add_test(NAME core.ascend.batchnorm_inference_host_oracle
    COMMAND flagdnn_test_ascend_batchnorm_inference_oracle)
  set_tests_properties(core.ascend.batchnorm_inference_host_oracle PROPERTIES
    LABELS "core;ascend;no_device" TIMEOUT 30)

  add_executable(flagdnn_test_ascend_embedded_python_environment
    tests/embedded_python_environment_test.cpp
    engines/embedded_python.cpp)
  target_compile_features(flagdnn_test_ascend_embedded_python_environment
    PRIVATE cxx_std_20)
  target_include_directories(flagdnn_test_ascend_embedded_python_environment
    PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(flagdnn_test_ascend_embedded_python_environment
    PRIVATE Python::Python)
  flagdnn_enable_warnings(flagdnn_test_ascend_embedded_python_environment)
  add_test(NAME core.ascend.embedded_python_environment
    COMMAND flagdnn_test_ascend_embedded_python_environment
      "${Python_EXECUTABLE}"
      "${_flagdnn_ascend_python_module_root}")
  set_tests_properties(core.ascend.embedded_python_environment PROPERTIES
    LABELS "core;ascend;no_device" TIMEOUT 30)

  add_executable(flagdnn_test_ascend_reduction_oracle
    tests/reduction_oracle_test.cpp
    engines/reduction_oracle.cpp)
  target_compile_features(flagdnn_test_ascend_reduction_oracle PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_reduction_oracle PRIVATE
    ${PROJECT_SOURCE_DIR})
  flagdnn_enable_warnings(flagdnn_test_ascend_reduction_oracle)
  add_test(NAME core.ascend.reduction_host_oracle
    COMMAND flagdnn_test_ascend_reduction_oracle)
  set_tests_properties(core.ascend.reduction_host_oracle PROPERTIES
    LABELS "core;ascend;no_device" TIMEOUT 30)

  add_executable(flagdnn_test_ascend_matmul_oracle
    tests/matmul_oracle_test.cpp
    engines/matmul_oracle.cpp)
  target_compile_features(flagdnn_test_ascend_matmul_oracle PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_matmul_oracle PRIVATE
    ${PROJECT_SOURCE_DIR})
  flagdnn_enable_warnings(flagdnn_test_ascend_matmul_oracle)
  add_test(NAME core.ascend.matmul_host_oracle
    COMMAND flagdnn_test_ascend_matmul_oracle)
  set_tests_properties(core.ascend.matmul_host_oracle PROPERTIES
    LABELS "core;ascend;no_device" TIMEOUT 30)

  add_executable(flagdnn_test_ascend_convolution_fprop_oracle
    tests/convolution_fprop_oracle_test.cpp
    engines/convolution_fprop_oracle.cpp)
  target_compile_features(flagdnn_test_ascend_convolution_fprop_oracle PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_convolution_fprop_oracle PRIVATE
    ${PROJECT_SOURCE_DIR})
  flagdnn_enable_warnings(flagdnn_test_ascend_convolution_fprop_oracle)
  add_test(NAME core.ascend.convolution_fprop_host_oracle
    COMMAND flagdnn_test_ascend_convolution_fprop_oracle)
  set_tests_properties(core.ascend.convolution_fprop_host_oracle PROPERTIES
    LABELS "core;ascend;no_device" TIMEOUT 30)

  add_executable(flagdnn_test_ascend_domain
    tests/domain_state_test.cpp
    context.cpp
    error.cpp
    engines/embedded_python.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp)
  target_compile_features(flagdnn_test_ascend_domain PRIVATE cxx_std_20)
  target_include_directories(flagdnn_test_ascend_domain PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src)
  target_include_directories(flagdnn_test_ascend_domain SYSTEM PRIVATE
    "${_flagdnn_ascend_ltj_include}"
    "${_flagdnn_ascend_arch_root}/include/experiment/msprof/toolchain"
    "${_flagdnn_ascend_runtime_header_root}"
    "${_flagdnn_ascend_runtime_header_root}/runtime/runtime")
  target_compile_definitions(flagdnn_test_ascend_domain PRIVATE
    ${_flagdnn_ascend_identity_definitions})
  target_link_libraries(flagdnn_test_ascend_domain PRIVATE
    Ascend::ascendcl Ascend::runtime Python::Python Threads::Threads
    ${CMAKE_DL_LIBS})
  flagdnn_enable_warnings(flagdnn_test_ascend_domain)
  add_test(NAME core.ascend.domain_state
    COMMAND flagdnn_test_ascend_domain)
  set_tests_properties(core.ascend.domain_state PROPERTIES
    LABELS "core;ascend;no_device" TIMEOUT 60)

  add_executable(flagdnn_test_ascend_python_stdout_containment
    tests/python_stdout_containment_test.cpp
    engines/python_stdout_containment.cpp)
  target_compile_features(flagdnn_test_ascend_python_stdout_containment
    PRIVATE cxx_std_20)
  target_include_directories(flagdnn_test_ascend_python_stdout_containment
    PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(flagdnn_test_ascend_python_stdout_containment
    PRIVATE Python::Python)
  flagdnn_enable_warnings(flagdnn_test_ascend_python_stdout_containment)
  add_test(NAME core.ascend.python_stdout_containment
    COMMAND flagdnn_test_ascend_python_stdout_containment)
  set_tests_properties(core.ascend.python_stdout_containment PROPERTIES
    LABELS "core;ascend;no_device" TIMEOUT 30)

  add_test(NAME core.ascend.no_owned_acl_lifecycle
    COMMAND "${CMAKE_COMMAND}"
      -DBACKEND_LIBRARY=$<TARGET_FILE:flagdnn_backend_ascend>
      -DLTJ_LIBRARY=${_flagdnn_ascend_triton_jit_library}
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/VerifyNoOwnedAcl.cmake")
  set_tests_properties(core.ascend.no_owned_acl_lifecycle PROPERTIES
    LABELS "core;ascend;no_device")

  add_test(NAME core.ascend.workspace_attestation_contract
    COMMAND "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/workspace_attestation_contract_test.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/workspace_attestation.py")
  set_tests_properties(core.ascend.workspace_attestation_contract PROPERTIES
    LABELS "core;ascend;no_device;production;attestation"
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1;PYTHONNOUSERSITE=1"
    TIMEOUT 60)

  add_executable(flagdnn_test_ascend_production_preflight
    tests/production_preflight_test.cpp)
  target_compile_features(flagdnn_test_ascend_production_preflight PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_production_preflight PRIVATE
    ${PROJECT_SOURCE_DIR})
  target_link_libraries(flagdnn_test_ascend_production_preflight PRIVATE
    ${CMAKE_DL_LIBS})
  flagdnn_enable_warnings(flagdnn_test_ascend_production_preflight)
  add_test(NAME core.ascend.production_preflight_fail_fast
    COMMAND flagdnn_test_ascend_production_preflight
      $<TARGET_FILE:flagdnn_backend_ascend>
      "${CMAKE_CURRENT_BINARY_DIR}/flagdnn-production-preflight-must-not-create")
  set_tests_properties(core.ascend.production_preflight_fail_fast PROPERTIES
    LABELS "core;ascend;no_device" TIMEOUT 60)

  add_test(NAME integration.ascend.installed_consumer_contract
    COMMAND "${CMAKE_COMMAND}"
      "-DPROJECT_BUILD_DIR=${PROJECT_BINARY_DIR}"
      "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
      "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/installed-consumer-contract"
      "-DBUILD_CONFIG=$<CONFIG>"
      "-DCANN_ROOT=${FLAGDNN_ASCEND_CANN_ROOT}"
      "-DCANN_VERSION=${FLAGDNN_ASCEND_CANN_VERSION}"
      "-DCODEGEN_PYTHON=${Python_EXECUTABLE}"
      "-DPYTHON_MODULE_ROOT=${_flagdnn_ascend_python_module_root}"
      "-DLTJ_LIBRARY=${_flagdnn_ascend_triton_jit_library}"
      "-DLTJ_STANDALONE=${_flagdnn_ascend_ltj_standalone}"
      "-DLTJ_CONFIG=${_flagdnn_ascend_ltj_config}"
      "-DLTJ_INCLUDE=${_flagdnn_ascend_ltj_include}"
      "-DRESOURCE_INSTALL_DIR=${FLAGDNN_RESOURCE_INSTALL_DIR}"
      "-DLIB_INSTALL_DIR=${CMAKE_INSTALL_LIBDIR}"
      -P "${PROJECT_SOURCE_DIR}/tests/core/VerifyInstalledAscendConsumer.cmake")
  set_tests_properties(integration.ascend.installed_consumer_contract PROPERTIES
    LABELS "integration;ascend;no_device;installed" RUN_SERIAL TRUE TIMEOUT 120)

  add_executable(flagdnn_test_ascend_embedded_preflight
    tests/embedded_preflight_test.cpp)
  target_compile_features(flagdnn_test_ascend_embedded_preflight PRIVATE
    cxx_std_20)
  target_link_libraries(flagdnn_test_ascend_embedded_preflight PRIVATE
    ${_flagdnn_ascend_triton_jit_target} Python::Python)
  get_filename_component(_flagdnn_ascend_site_packages
    "${Torch_DIR}/../../../.." ABSOLUTE)
  set(_flagdnn_ascend_torch_private_libraries
    "${_flagdnn_ascend_site_packages}/torch.libs")
  if(IS_DIRECTORY "${_flagdnn_ascend_torch_private_libraries}")
    target_link_options(flagdnn_test_ascend_embedded_preflight PRIVATE
      "LINKER:-rpath-link,${_flagdnn_ascend_torch_private_libraries}")
    set_property(TARGET flagdnn_test_ascend_embedded_preflight APPEND PROPERTY
      BUILD_RPATH "${_flagdnn_ascend_torch_private_libraries}")
  endif()
  get_filename_component(_flagdnn_ascend_ltj_library_directory
    "${_flagdnn_ascend_triton_jit_library}" DIRECTORY)
  get_filename_component(_flagdnn_ascend_python_library_directory
    "${_flagdnn_ascend_python_library}" DIRECTORY)
  get_filename_component(_flagdnn_ascend_ascendcl_library_directory
    "${FLAGDNN_ASCEND_ASCENDCL_LIBRARY}" DIRECTORY)
  get_filename_component(_flagdnn_ascend_runtime_library_directory
    "${FLAGDNN_ASCEND_RUNTIME_LIBRARY}" DIRECTORY)
  set(_flagdnn_ascend_embedded_library_directories
    "${_flagdnn_ascend_ltj_library_directory}"
    "${_flagdnn_ascend_python_library_directory}"
    "${_flagdnn_ascend_ascendcl_library_directory}"
    "${_flagdnn_ascend_runtime_library_directory}"
    "${FLAGDNN_ASCEND_CANN_ROOT}/lib64"
    "${_flagdnn_ascend_site_packages}/torch/lib")
  set(_flagdnn_ascend_driver_root "")
  if(DEFINED ENV{ASCEND_DRIVER_HOME} AND
     IS_DIRECTORY "$ENV{ASCEND_DRIVER_HOME}")
    file(REAL_PATH "$ENV{ASCEND_DRIVER_HOME}"
      _flagdnn_ascend_driver_root)
  elseif(IS_DIRECTORY "/usr/local/Ascend/driver")
    file(REAL_PATH "/usr/local/Ascend/driver"
      _flagdnn_ascend_driver_root)
  endif()
  if(_flagdnn_ascend_driver_root)
    foreach(relative IN ITEMS lib64 lib64/common lib64/driver)
      if(IS_DIRECTORY "${_flagdnn_ascend_driver_root}/${relative}")
        list(APPEND _flagdnn_ascend_embedded_library_directories
          "${_flagdnn_ascend_driver_root}/${relative}")
      endif()
    endforeach()
  endif()
  if(IS_DIRECTORY "${_flagdnn_ascend_torch_private_libraries}")
    list(APPEND _flagdnn_ascend_embedded_library_directories
      "${_flagdnn_ascend_torch_private_libraries}")
  endif()
  list(REMOVE_DUPLICATES _flagdnn_ascend_embedded_library_directories)
  foreach(directory IN LISTS _flagdnn_ascend_embedded_library_directories)
    if(NOT IS_DIRECTORY "${directory}")
      message(FATAL_ERROR
        "Embedded Ascend preflight library directory is missing: ${directory}")
    endif()
  endforeach()
  list(JOIN _flagdnn_ascend_embedded_library_directories ":"
    _flagdnn_ascend_embedded_library_path)

  foreach(_flagdnn_ascend_offline_kernel IN ITEMS
      batchnorm_inference batchnorm_training convolution_fprop
      layernorm matmul rmsnorm)
    add_test(
      NAME integration.ascend.offline_compile.${_flagdnn_ascend_offline_kernel}
      COMMAND "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/${_flagdnn_ascend_offline_kernel}_offline_test.py")
    set_tests_properties(
      integration.ascend.offline_compile.${_flagdnn_ascend_offline_kernel}
      PROPERTIES
        LABELS "integration;ascend;no_device;compiler;offline"
        RUN_SERIAL TRUE
        TIMEOUT 1800
        ENVIRONMENT
          "ASCEND_HOME_PATH=${FLAGDNN_ASCEND_CANN_ROOT};ASCEND_TOOLKIT_HOME=${FLAGDNN_ASCEND_CANN_ROOT};FLAGDNN_STANDALONE_COMPILE=${_flagdnn_ascend_ltj_standalone};LD_LIBRARY_PATH=${_flagdnn_ascend_embedded_library_path};PYTHONDONTWRITEBYTECODE=1;PYTHONNOUSERSITE=1;TORCH_DEVICE_BACKEND_AUTOLOAD=0")
  endforeach()

  target_include_directories(flagdnn_test_ascend_embedded_preflight SYSTEM
    PRIVATE
      "${FLAGDNN_ASCEND_CANN_ROOT}/include"
      "${FLAGDNN_ASCEND_CANN_ROOT}/include/aclnn"
      "${FLAGDNN_ASCEND_CANN_ROOT}/include/experiment"
      "${FLAGDNN_ASCEND_CANN_ROOT}/include/experiment/runtime"
      "${_flagdnn_ascend_arch_root}/include"
      "${_flagdnn_ascend_arch_root}/include/experiment"
      "${_flagdnn_ascend_arch_root}/include/experiment/msprof"
      "${_flagdnn_ascend_runtime_header_root}"
      "${_flagdnn_ascend_runtime_header_root}/runtime/runtime")
  flagdnn_enable_warnings(flagdnn_test_ascend_embedded_preflight)
  add_test(NAME integration.ascend.embedded_toolchain_preflight
    COMMAND flagdnn_test_ascend_embedded_preflight
      "${CMAKE_CURRENT_BINARY_DIR}/embedded-preflight-cache"
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/binary.py"
      "${_flagdnn_ascend_ltj_standalone}"
      "${CMAKE_CURRENT_SOURCE_DIR}/workspace_attestation.py"
      "${_flagdnn_ascend_python_module_root}")
  set_tests_properties(integration.ascend.embedded_toolchain_preflight PROPERTIES
    LABELS "integration;ascend;no_device;toolchain" RUN_SERIAL TRUE TIMEOUT 120
    ENVIRONMENT
      "ASCEND_HOME_PATH=${FLAGDNN_ASCEND_CANN_ROOT};LD_LIBRARY_PATH=${_flagdnn_ascend_embedded_library_path};PYTHONDONTWRITEBYTECODE=1;PYTHONNOUSERSITE=1")

  if(_flagdnn_ascend_driver_root)
    add_test(NAME integration.ascend.installed_consumer_add
      COMMAND "${CMAKE_COMMAND}"
        "-DPROJECT_BUILD_DIR=${PROJECT_BINARY_DIR}"
        "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
        "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/installed-consumer-device-trusted-local"
        "-DBUILD_CONFIG=$<CONFIG>"
        "-DCANN_ROOT=${FLAGDNN_ASCEND_CANN_ROOT}"
        "-DCANN_VERSION=${FLAGDNN_ASCEND_CANN_VERSION}"
        "-DCODEGEN_PYTHON=${Python_EXECUTABLE}"
        "-DPYTHON_LIBRARY=${_flagdnn_ascend_python_library}"
        "-DPYTHON_MODULE_ROOT=${_flagdnn_ascend_python_module_root}"
        "-DLTJ_LIBRARY=${_flagdnn_ascend_triton_jit_library}"
        "-DLTJ_STANDALONE=${_flagdnn_ascend_ltj_standalone}"
        "-DLTJ_CONFIG=${_flagdnn_ascend_ltj_config}"
        "-DLTJ_INCLUDE=${_flagdnn_ascend_ltj_include}"
        "-DASCEND_DRIVER_ROOT=${_flagdnn_ascend_driver_root}"
        "-DRESOURCE_INSTALL_DIR=${FLAGDNN_RESOURCE_INSTALL_DIR}"
        "-DLIB_INSTALL_DIR=${CMAKE_INSTALL_LIBDIR}"
        -DRUN_ASCEND_DEVICE_TEST=ON
        -P "${PROJECT_SOURCE_DIR}/tests/core/VerifyInstalledAscendConsumer.cmake")
    set_tests_properties(
      integration.ascend.installed_consumer_add PROPERTIES
      LABELS "integration;ascend;runtime;device;installed;trusted_local"
      RUN_SERIAL TRUE TIMEOUT 720)
  endif()

  add_executable(flagdnn_test_ascend_artifact_contract
    tests/artifact_contract_driver.cpp
    artifact.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/json.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp)
  target_compile_features(flagdnn_test_ascend_artifact_contract PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_artifact_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/generated/include
    ${FLAGDNN_ASCEND_INCLUDE_DIR})
  flagdnn_enable_warnings(flagdnn_test_ascend_artifact_contract)
  add_test(NAME core.ascend.artifact_graph_contract
    COMMAND "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/artifact_contract_test.py"
      $<TARGET_FILE:flagdnn_test_ascend_artifact_contract>
      "ascend_Ascend910B1_cann_${FLAGDNN_ASCEND_CANN_VERSION}_aic_24")
  set_tests_properties(core.ascend.artifact_graph_contract PROPERTIES
    LABELS "core;ascend;no_device"
    ENVIRONMENT
      "PYTHONDONTWRITEBYTECODE=1;PYTHONNOUSERSITE=1;TRITON_JIT_BACKEND=NPU;TRITON_BACKEND=torch_npu;TORCH_DEVICE_BACKEND_AUTOLOAD=0;TRITON_ASCEND_ARCH=Ascend910B1"
    TIMEOUT 60)
endif()
