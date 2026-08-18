# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

# Ascend validation is deliberately isolated from the production plugin. Only
# targets in this directory may include ACLNN headers or link operator API
# libraries.
include("${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")

if(NOT TARGET flagdnn_backend_ascend)
  message(FATAL_ERROR "Ascend validation requires FLAGDNN_BACKENDS=ascend")
endif()
if(NOT TARGET Ascend::ascendcl OR NOT TARGET Ascend::runtime)
  message(FATAL_ERROR
    "Ascend validation requires the production AscendCL imported targets")
endif()
if(NOT CANN_ROOT OR NOT IS_DIRECTORY "${CANN_ROOT}")
  message(FATAL_ERROR "Ascend validation requires a valid CANN_ROOT")
endif()
if(NOT FLAGDNN_CODEGEN_PYTHON)
  message(FATAL_ERROR "Ascend validation requires FLAGDNN_CODEGEN_PYTHON")
endif()

execute_process(
  COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -I -c
    "import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()['purelib']).resolve())"
  RESULT_VARIABLE _flagdnn_validation_python_root_status
  OUTPUT_VARIABLE _flagdnn_validation_python_module_root
  ERROR_VARIABLE _flagdnn_validation_python_root_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
  TIMEOUT 10)
if(NOT _flagdnn_validation_python_root_status EQUAL 0 OR
   NOT IS_DIRECTORY "${_flagdnn_validation_python_module_root}")
  message(FATAL_ERROR
    "Ascend validation cannot resolve the codegen Python module root: "
    "${_flagdnn_validation_python_root_error}")
endif()
file(REAL_PATH "${_flagdnn_validation_python_module_root}"
  _flagdnn_validation_python_module_root)

file(REAL_PATH "${CANN_ROOT}" _flagdnn_validation_cann_root)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(_flagdnn_validation_arch_directory "aarch64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_flagdnn_validation_arch_directory "x86_64-linux")
else()
  message(FATAL_ERROR
    "Ascend validation does not support ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(_flagdnn_validation_arch_root
  "${_flagdnn_validation_cann_root}/${_flagdnn_validation_arch_directory}")
find_path(_flagdnn_validation_aclnn_include
  NAMES
    aclnn/acl_meta.h
    aclnnop/aclnn_batch_norm.h
    aclnnop/aclnn_batch_norm_elemt.h
    aclnnop/aclnn_add.h
    aclnnop/aclnn_sub.h
    aclnnop/aclnn_mul.h
    aclnnop/aclnn_div.h
    aclnnop/aclnn_eq_tensor.h
    aclnnop/aclnn_fmod_tensor.h
    aclnnop/aclnn_ge_tensor.h
    aclnnop/aclnn_gt_tensor.h
    aclnnop/aclnn_ne_tensor.h
    aclnnop/aclnn_pow_tensor_tensor.h
    aclnnop/aclnn_s_where.h
    aclnnop/aclnn_flatten.h
    aclnnop/aclnn_permute.h
    aclnnop/aclnn_slice_v2.h
    aclnnop/aclnn_minimum.h
    aclnnop/aclnn_maximum.h
    aclnnop/aclnn_abs.h
    aclnnop/aclnn_ceil.h
    aclnnop/aclnn_clamp.h
    aclnnop/aclnn_cos.h
    aclnnop/aclnn_elu.h
    aclnnop/aclnn_erf.h
    aclnnop/aclnn_exp.h
    aclnnop/aclnn_floor.h
    aclnnop/aclnn_gelu.h
    aclnnop/aclnn_gelu_v2.h
    aclnnop/aclnn_log.h
    aclnnop/aclnn_logical_and.h
    aclnnop/aclnn_logical_not.h
    aclnnop/aclnn_logical_or.h
    aclnnop/aclnn_matmul.h
    aclnnop/aclnn_convolution.h
    aclnnop/aclnn_neg.h
    aclnnop/aclnn_reciprocal.h
    aclnnop/aclnn_reduce_sum.h
    aclnnop/aclnn_mean.h
    aclnnop/aclnn_layer_norm.h
    aclnnop/aclnn_prod.h
    aclnnop/aclnn_rsqrt.h
    aclnnop/aclnn_sigmoid.h
    aclnnop/aclnn_sigmoid_backward.h
    aclnnop/aclnn_sin.h
    aclnnop/aclnn_softplus.h
    aclnnop/aclnn_sqrt.h
    aclnnop/aclnn_swish.h
    aclnnop/aclnn_tan.h
    aclnnop/aclnn_tanh.h
  PATHS
    "${_flagdnn_validation_arch_root}/include"
    "${_flagdnn_validation_arch_root}/pkg_inc"
  NO_DEFAULT_PATH
  NO_CACHE)
find_library(_flagdnn_validation_nnopbase
  NAMES nnopbase
  PATHS "${_flagdnn_validation_arch_root}/lib64"
  NO_DEFAULT_PATH
  NO_CACHE)
find_library(_flagdnn_validation_opapi_math
  NAMES opapi_math
  PATHS "${_flagdnn_validation_arch_root}/lib64"
  NO_DEFAULT_PATH
  NO_CACHE)
find_library(_flagdnn_validation_opapi_nn
  NAMES opapi_nn
  PATHS "${_flagdnn_validation_arch_root}/lib64"
  NO_DEFAULT_PATH
  NO_CACHE)
if(NOT _flagdnn_validation_aclnn_include OR
   NOT _flagdnn_validation_nnopbase OR
   NOT _flagdnn_validation_opapi_math OR
   NOT _flagdnn_validation_opapi_nn)
  message(FATAL_ERROR
    "Ascend validation requires ACLNN headers, libnnopbase.so, and "
    "the precise libopapi_math.so/libopapi_nn.so libraries from the "
    "selected CANN 9 installation")
endif()
if(NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnn/acl_meta.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_add.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_sub.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_mul.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_div.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_eq_tensor.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_ge_tensor.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_gt_tensor.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_ne_tensor.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_fmod_tensor.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_pow_tensor_tensor.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_s_where.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_minimum.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_maximum.h")
  message(FATAL_ERROR
    "Ascend validation requires binary pointwise ACLNN headers")
endif()
if(NOT EXISTS
    "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_batch_norm_elemt.h")
  message(FATAL_ERROR
    "Ascend BatchNorm inference validation requires the exact "
    "aclnn_batch_norm_elemt.h header")
endif()
if(NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_matmul.h")
  message(FATAL_ERROR
    "Ascend MatMul validation requires the exact aclnn_matmul.h header")
endif()
if(NOT EXISTS
    "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_convolution.h")
  message(FATAL_ERROR
    "Ascend convolution validation requires the exact "
    "aclnn_convolution.h header")
endif()
if(NOT EXISTS
    "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_batch_norm.h")
  message(FATAL_ERROR
    "Ascend BatchNorm validation requires the exact "
    "aclnn_batch_norm.h header")
endif()
if(NOT EXISTS
    "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_layer_norm.h")
  message(FATAL_ERROR
    "Ascend LayerNorm validation requires the exact "
    "aclnn_layer_norm.h header")
endif()
if(NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_abs.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_ceil.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_clamp.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_cos.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_erf.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_exp.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_floor.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_log.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_logical_and.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_logical_not.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_logical_or.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_neg.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_reciprocal.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_rsqrt.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_sin.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_sqrt.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_tan.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_tanh.h")
  message(FATAL_ERROR
    "Ascend validation requires unary ACLNN math primitive headers")
endif()
if(NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_reduce_sum.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_mean.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_prod.h")
  message(FATAL_ERROR
    "Ascend validation requires ACLNN reduction math primitive headers")
endif()
if(NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_flatten.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_permute.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_slice_v2.h")
  message(FATAL_ERROR
    "Ascend validation requires ACLNN layout primitive headers")
endif()
if(NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_elu.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_gelu.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_gelu_v2.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_sigmoid.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_sigmoid_backward.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_softplus.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_swish.h")
  message(FATAL_ERROR
    "Ascend validation requires unary ACLNN neural-network headers")
endif()
foreach(_flagdnn_validation_library IN ITEMS
    _flagdnn_validation_nnopbase _flagdnn_validation_opapi_math
    _flagdnn_validation_opapi_nn)
  file(REAL_PATH "${${_flagdnn_validation_library}}"
    "${_flagdnn_validation_library}")
endforeach()
get_filename_component(_flagdnn_validation_nnopbase_name
  "${_flagdnn_validation_nnopbase}" NAME)
get_filename_component(_flagdnn_validation_opapi_math_name
  "${_flagdnn_validation_opapi_math}" NAME)
get_filename_component(_flagdnn_validation_opapi_nn_name
  "${_flagdnn_validation_opapi_nn}" NAME)
if(NOT _flagdnn_validation_nnopbase_name MATCHES "^libnnopbase\\.so" OR
   NOT _flagdnn_validation_opapi_math_name MATCHES "^libopapi_math\\.so" OR
   NOT _flagdnn_validation_opapi_nn_name MATCHES "^libopapi_nn\\.so")
  message(FATAL_ERROR
    "Ascend validation resolved an imprecise ACLNN dependency set")
endif()

add_library(FlagDNNAscendValidation::nnopbase SHARED IMPORTED GLOBAL)
set_target_properties(FlagDNNAscendValidation::nnopbase PROPERTIES
  IMPORTED_LOCATION "${_flagdnn_validation_nnopbase}"
  INTERFACE_INCLUDE_DIRECTORIES "${_flagdnn_validation_aclnn_include}")
add_library(FlagDNNAscendValidation::opapi_math SHARED IMPORTED GLOBAL)
set_target_properties(FlagDNNAscendValidation::opapi_math PROPERTIES
  IMPORTED_LOCATION "${_flagdnn_validation_opapi_math}"
  INTERFACE_INCLUDE_DIRECTORIES "${_flagdnn_validation_aclnn_include}")
add_library(FlagDNNAscendValidation::opapi_nn SHARED IMPORTED GLOBAL)
set_target_properties(FlagDNNAscendValidation::opapi_nn PROPERTIES
  IMPORTED_LOCATION "${_flagdnn_validation_opapi_nn}"
  INTERFACE_INCLUDE_DIRECTORIES "${_flagdnn_validation_aclnn_include}")

function(_flagdnn_validation_imported_location target output)
  set(_flagdnn_validation_location "")
  foreach(_flagdnn_validation_property IN ITEMS
      IMPORTED_LOCATION IMPORTED_LOCATION_NOCONFIG IMPORTED_LOCATION_RELEASE
      IMPORTED_LOCATION_RELWITHDEBINFO IMPORTED_LOCATION_DEBUG)
    if(NOT _flagdnn_validation_location)
      get_target_property(_flagdnn_validation_candidate "${target}"
        "${_flagdnn_validation_property}")
      if(_flagdnn_validation_candidate AND
         EXISTS "${_flagdnn_validation_candidate}")
        file(REAL_PATH "${_flagdnn_validation_candidate}"
          _flagdnn_validation_location)
      endif()
    endif()
  endforeach()
  if(NOT _flagdnn_validation_location)
    message(FATAL_ERROR "Cannot determine imported location for ${target}")
  endif()
  set("${output}" "${_flagdnn_validation_location}" PARENT_SCOPE)
endfunction()

_flagdnn_validation_imported_location(
  Ascend::ascendcl _flagdnn_validation_ascendcl)
_flagdnn_validation_imported_location(
  Ascend::runtime _flagdnn_validation_runtime)
# Production promotes its selected imported target to global scope. Read that
# exact target so validation cannot accidentally hash a different installation.
if(TARGET TritonJIT::triton_jit)
  _flagdnn_validation_imported_location(
    TritonJIT::triton_jit _flagdnn_validation_triton_jit)
elseif(TARGET triton_jit)
  _flagdnn_validation_imported_location(
    triton_jit _flagdnn_validation_triton_jit)
else()
  message(FATAL_ERROR
    "Ascend validation cannot inspect the production NPU libtriton_jit target")
endif()

file(SHA256 "${_flagdnn_validation_ascendcl}"
  _flagdnn_validation_ascendcl_sha256)
file(SHA256 "${_flagdnn_validation_runtime}"
  _flagdnn_validation_runtime_sha256)
file(SHA256 "${_flagdnn_validation_triton_jit}"
  _flagdnn_validation_triton_jit_sha256)
file(SHA256 "${_flagdnn_validation_nnopbase}"
  _flagdnn_validation_nnopbase_sha256)
file(SHA256 "${_flagdnn_validation_opapi_math}"
  _flagdnn_validation_opapi_math_sha256)
file(SHA256 "${_flagdnn_validation_opapi_nn}"
  _flagdnn_validation_opapi_nn_sha256)

if(NOT CMAKE_READELF)
  find_program(CMAKE_READELF readelf REQUIRED)
endif()
function(_flagdnn_validation_object_id library sha256 output)
  execute_process(COMMAND "${CMAKE_READELF}" -n "${library}"
    RESULT_VARIABLE _flagdnn_build_id_status
    OUTPUT_VARIABLE _flagdnn_build_id_notes
    ERROR_VARIABLE _flagdnn_build_id_error)
  if(NOT _flagdnn_build_id_status EQUAL 0)
    message(FATAL_ERROR
      "Cannot inspect build ID for ${library}: ${_flagdnn_build_id_error}")
  endif()
  string(REGEX MATCH "Build ID: ([0-9A-Fa-f]+)"
    _flagdnn_build_id_match "${_flagdnn_build_id_notes}")
  if(CMAKE_MATCH_1)
    string(TOLOWER "${CMAKE_MATCH_1}" _flagdnn_build_id_value)
    set(_flagdnn_build_id_value "build-id:${_flagdnn_build_id_value}")
  else()
    set(_flagdnn_build_id_value "sha256:${sha256}")
  endif()
  set("${output}" "${_flagdnn_build_id_value}" PARENT_SCOPE)
endfunction()
_flagdnn_validation_object_id("${_flagdnn_validation_ascendcl}"
  "${_flagdnn_validation_ascendcl_sha256}"
  _flagdnn_validation_ascendcl_build_id)
_flagdnn_validation_object_id("${_flagdnn_validation_runtime}"
  "${_flagdnn_validation_runtime_sha256}"
  _flagdnn_validation_runtime_build_id)

set(_flagdnn_validation_install_info
  "${_flagdnn_validation_arch_root}/ascend_toolkit_install.info")
if(NOT EXISTS "${_flagdnn_validation_install_info}")
  message(FATAL_ERROR "Ascend validation cannot find CANN install metadata")
endif()
file(STRINGS "${_flagdnn_validation_install_info}"
  _flagdnn_validation_version_line REGEX "^version=")
list(LENGTH _flagdnn_validation_version_line
  _flagdnn_validation_version_count)
if(NOT _flagdnn_validation_version_count EQUAL 1)
  message(FATAL_ERROR "CANN install metadata has no unique version entry")
endif()
string(REGEX REPLACE "^version=" "" _flagdnn_validation_cann_version
  "${_flagdnn_validation_version_line}")

add_library(flagdnn_validation_ascend_platform INTERFACE)
target_compile_features(flagdnn_validation_ascend_platform
  INTERFACE cxx_std_20)
target_include_directories(flagdnn_validation_ascend_platform INTERFACE
  "${_flagdnn_validation_aclnn_include}"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_link_libraries(flagdnn_validation_ascend_platform INTERFACE
  Ascend::ascendcl Ascend::runtime)

add_library(flagdnn_validation_ascend_tensor_io STATIC tensor_io.cpp)
target_compile_features(flagdnn_validation_ascend_tensor_io PUBLIC cxx_std_20)
target_link_libraries(flagdnn_validation_ascend_tensor_io PUBLIC
  FlagDNN::flagdnn flagdnn_validation_ascend_platform)
flagdnn_enable_warnings(flagdnn_validation_ascend_tensor_io)

add_library(flagdnn_validation_ascend_aclnn INTERFACE)
target_link_libraries(flagdnn_validation_ascend_aclnn INTERFACE
  FlagDNNAscendValidation::nnopbase
  FlagDNNAscendValidation::opapi_math)

add_library(flagdnn_validation_ascend_aclnn_nn INTERFACE)
target_link_libraries(flagdnn_validation_ascend_aclnn_nn INTERFACE
  flagdnn_validation_ascend_aclnn
  FlagDNNAscendValidation::opapi_nn)

get_filename_component(_flagdnn_validation_library_directory
  "${_flagdnn_validation_opapi_math}" DIRECTORY)

if(FLAGDNN_BUILD_TESTS)
  find_package(Threads REQUIRED)
  add_executable(flagdnn_test_ascend_exact_reference_contract
    exact_reference_contract_test.cpp)
  target_compile_features(
    flagdnn_test_ascend_exact_reference_contract PRIVATE cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_exact_reference_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  flagdnn_enable_warnings(flagdnn_test_ascend_exact_reference_contract)
  add_test(NAME core.ascend.exact_reference_contract
    COMMAND flagdnn_test_ascend_exact_reference_contract)
  set_tests_properties(core.ascend.exact_reference_contract PROPERTIES
    LABELS "core;ascend;no_device;validation"
    TIMEOUT 60)

  add_executable(flagdnn_test_ascend_layernorm_contract
    layernorm_contract_test.cpp
    layernorm_validation.cpp
    benchmark/aclnn_layernorm_provider.cpp
    functional/aclnn_layernorm.cpp
    functional/aclnn_layernorm_plan.cpp
    "${PROJECT_SOURCE_DIR}/tests/common/normalization.cpp")
  target_compile_features(
    flagdnn_test_ascend_layernorm_contract PRIVATE cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_layernorm_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_link_libraries(
    flagdnn_test_ascend_layernorm_contract PRIVATE
    flagdnn_benchmark_cases
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn
    FlagDNN::flagdnn)
  flagdnn_enable_warnings(flagdnn_test_ascend_layernorm_contract)
  add_test(NAME core.ascend.layernorm_validation_contract
    COMMAND flagdnn_test_ascend_layernorm_contract)
  set_tests_properties(core.ascend.layernorm_validation_contract PROPERTIES
    LABELS "core;ascend;no_device;validation;normalization"
    TIMEOUT 60)

  add_library(flagdnn_test_ascend_layernorm_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    layernorm_validation.cpp
    functional/aclnn_layernorm.cpp
    functional/aclnn_layernorm_plan.cpp
    functional/layernorm_runner.cpp)
  target_compile_features(
    flagdnn_test_ascend_layernorm_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_layernorm_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(
    flagdnn_test_ascend_layernorm_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(
    flagdnn_test_ascend_layernorm_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn)
  flagdnn_enable_warnings(flagdnn_test_ascend_layernorm_adapter)

  flagdnn_register_functional_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_test_ascend_layernorm_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS layernorm
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;device;normalization"
    TIMEOUT 1200
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  add_executable(flagdnn_test_ascend_rmsnorm_contract
    rmsnorm_contract_test.cpp
    rmsnorm_validation.cpp
    benchmark/aclnn_rmsnorm_provider.cpp
    functional/aclnn_rmsnorm.cpp
    functional/aclnn_rmsnorm_plan.cpp
    "${PROJECT_SOURCE_DIR}/tests/common/normalization.cpp")
  target_compile_features(
    flagdnn_test_ascend_rmsnorm_contract PRIVATE cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_rmsnorm_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_link_libraries(
    flagdnn_test_ascend_rmsnorm_contract PRIVATE
    flagdnn_benchmark_cases
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn
    FlagDNN::flagdnn)
  flagdnn_enable_warnings(flagdnn_test_ascend_rmsnorm_contract)
  add_test(NAME core.ascend.rmsnorm_validation_contract
    COMMAND flagdnn_test_ascend_rmsnorm_contract)
  set_tests_properties(core.ascend.rmsnorm_validation_contract PROPERTIES
    LABELS "core;ascend;no_device;validation;normalization"
    TIMEOUT 60)

  add_library(flagdnn_test_ascend_rmsnorm_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    rmsnorm_validation.cpp
    functional/aclnn_rmsnorm.cpp
    functional/aclnn_rmsnorm_plan.cpp
    functional/rmsnorm_runner.cpp)
  target_compile_features(
    flagdnn_test_ascend_rmsnorm_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_rmsnorm_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(
    flagdnn_test_ascend_rmsnorm_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(
    flagdnn_test_ascend_rmsnorm_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn)
  flagdnn_enable_warnings(flagdnn_test_ascend_rmsnorm_adapter)

  flagdnn_register_functional_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_test_ascend_rmsnorm_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS rmsnorm
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;device;normalization"
    TIMEOUT 1200
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  add_library(flagdnn_test_ascend_batchnorm_inference_adapter STATIC
    "${PROJECT_SOURCE_DIR}/tests/common/normalization.cpp"
    batchnorm_inference_validation.cpp
    functional/aclnn_batchnorm_inference.cpp
    functional/normalization_runner.cpp)
  target_compile_features(
    flagdnn_test_ascend_batchnorm_inference_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_batchnorm_inference_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(
    flagdnn_test_ascend_batchnorm_inference_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(
    flagdnn_test_ascend_batchnorm_inference_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn)
  flagdnn_enable_warnings(
    flagdnn_test_ascend_batchnorm_inference_adapter)

  flagdnn_register_functional_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_test_ascend_batchnorm_inference_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS batchnorm_inference
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;device;normalization"
    TIMEOUT 1200
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  add_executable(flagdnn_test_ascend_batchnorm_inference_contract
    batchnorm_inference_contract_test.cpp
    batchnorm_inference_validation.cpp
    functional/aclnn_batchnorm_inference.cpp
    "${PROJECT_SOURCE_DIR}/tests/common/normalization.cpp")
  target_compile_features(
    flagdnn_test_ascend_batchnorm_inference_contract PRIVATE cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_batchnorm_inference_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_link_libraries(
    flagdnn_test_ascend_batchnorm_inference_contract PRIVATE
    flagdnn_benchmark_cases
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn
    FlagDNN::flagdnn)
  flagdnn_enable_warnings(
    flagdnn_test_ascend_batchnorm_inference_contract)
  add_test(NAME core.ascend.batchnorm_inference_validation_contract
    COMMAND flagdnn_test_ascend_batchnorm_inference_contract)
  set_tests_properties(
    core.ascend.batchnorm_inference_validation_contract PROPERTIES
    LABELS "core;ascend;no_device;validation;normalization"
    TIMEOUT 60)

  add_executable(flagdnn_test_ascend_batchnorm_contract
    batchnorm_contract_test.cpp
    batchnorm_validation.cpp
    benchmark/aclnn_batchnorm_provider.cpp
    functional/aclnn_batchnorm.cpp
    "${PROJECT_SOURCE_DIR}/tests/common/normalization.cpp")
  target_compile_features(
    flagdnn_test_ascend_batchnorm_contract PRIVATE cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_batchnorm_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_link_libraries(
    flagdnn_test_ascend_batchnorm_contract PRIVATE
    flagdnn_benchmark_cases
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn
    FlagDNN::flagdnn)
  flagdnn_enable_warnings(flagdnn_test_ascend_batchnorm_contract)
  add_test(NAME core.ascend.batchnorm_validation_contract
    COMMAND flagdnn_test_ascend_batchnorm_contract)
  set_tests_properties(core.ascend.batchnorm_validation_contract PROPERTIES
    LABELS "core;ascend;no_device;validation;normalization"
    TIMEOUT 60)

  add_library(flagdnn_test_ascend_batchnorm_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    batchnorm_validation.cpp
    functional/aclnn_batchnorm.cpp
    functional/batchnorm_runner.cpp)
  target_compile_features(
    flagdnn_test_ascend_batchnorm_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_batchnorm_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(
    flagdnn_test_ascend_batchnorm_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(
    flagdnn_test_ascend_batchnorm_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn)
  flagdnn_enable_warnings(flagdnn_test_ascend_batchnorm_adapter)

  flagdnn_register_functional_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_test_ascend_batchnorm_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS batchnorm
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;device;normalization"
    TIMEOUT 1200
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  add_executable(flagdnn_test_ascend_gelu_contract
    gelu_contract_test.cpp)
  target_compile_features(flagdnn_test_ascend_gelu_contract PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_gelu_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  flagdnn_enable_warnings(flagdnn_test_ascend_gelu_contract)
  add_test(NAME core.ascend.gelu_validation_contract
    COMMAND flagdnn_test_ascend_gelu_contract)
  set_tests_properties(core.ascend.gelu_validation_contract PROPERTIES
    LABELS "core;ascend;no_device;validation"
    TIMEOUT 60)

  add_executable(flagdnn_test_ascend_mod_contract
    mod_contract_test.cpp)
  target_compile_features(flagdnn_test_ascend_mod_contract PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_mod_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_link_libraries(flagdnn_test_ascend_mod_contract PRIVATE
    flagdnn_validation_ascend_tensor_io)
  flagdnn_enable_warnings(flagdnn_test_ascend_mod_contract)
  add_test(NAME core.ascend.mod_validation_contract
    COMMAND flagdnn_test_ascend_mod_contract)
  set_tests_properties(core.ascend.mod_validation_contract PROPERTIES
    LABELS "core;ascend;no_device;validation"
    TIMEOUT 60)

  add_executable(flagdnn_test_ascend_sigmoid_backward_contract
    sigmoid_backward_contract_test.cpp)
  target_compile_features(
    flagdnn_test_ascend_sigmoid_backward_contract PRIVATE cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_sigmoid_backward_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  flagdnn_enable_warnings(flagdnn_test_ascend_sigmoid_backward_contract)
  add_test(NAME core.ascend.sigmoid_backward_validation_contract
    COMMAND flagdnn_test_ascend_sigmoid_backward_contract)
  set_tests_properties(
    core.ascend.sigmoid_backward_validation_contract PROPERTIES
    LABELS "core;ascend;no_device;validation"
    TIMEOUT 60)

  add_executable(flagdnn_test_ascend_tensor_io_contract
    tensor_io_contract_test.cpp)
  target_compile_features(flagdnn_test_ascend_tensor_io_contract PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_tensor_io_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_link_libraries(flagdnn_test_ascend_tensor_io_contract PRIVATE
    flagdnn_validation_ascend_tensor_io)
  flagdnn_enable_warnings(flagdnn_test_ascend_tensor_io_contract)
  add_test(NAME core.ascend.tensor_io.boolean_contract
    COMMAND flagdnn_test_ascend_tensor_io_contract)
  set_tests_properties(core.ascend.tensor_io.boolean_contract PROPERTIES
    LABELS "core;ascend;no_device;validation"
    TIMEOUT 60)

  add_library(flagdnn_test_ascend_reduction_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    functional/aclnn_reduction.cpp
    functional/aclnn_reduction_plan.cpp
    functional/reduction_runner.cpp
    reduction_validation.cpp)
  target_compile_features(flagdnn_test_ascend_reduction_adapter PUBLIC
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_reduction_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(flagdnn_test_ascend_reduction_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(flagdnn_test_ascend_reduction_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn)
  flagdnn_enable_warnings(flagdnn_test_ascend_reduction_adapter)

  add_executable(flagdnn_test_ascend_reduction_contract
    reduction_contract_test.cpp
    benchmark/aclnn_reduction_plan.cpp)
  target_compile_features(flagdnn_test_ascend_reduction_contract PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_reduction_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/benchmark")
  target_link_libraries(flagdnn_test_ascend_reduction_contract PRIVATE
    flagdnn_benchmark_cases
    flagdnn_test_ascend_reduction_adapter)
  flagdnn_enable_warnings(flagdnn_test_ascend_reduction_contract)
  add_test(NAME core.ascend.reduction_validation_contract
    COMMAND flagdnn_test_ascend_reduction_contract)
  set_tests_properties(core.ascend.reduction_validation_contract PROPERTIES
    LABELS "core;ascend;no_device;validation;reduction"
    TIMEOUT 60)

  add_executable(flagdnn_test_ascend_matmul_contract
    matmul_contract_test.cpp
    benchmark/aclnn_matmul_provider.cpp
    functional/aclnn_matmul.cpp
    functional/aclnn_matmul_plan.cpp
    "${PROJECT_SOURCE_DIR}/tests/common/matmul.cpp")
  target_compile_features(flagdnn_test_ascend_matmul_contract PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_matmul_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_link_libraries(flagdnn_test_ascend_matmul_contract PRIVATE
    flagdnn_benchmark_cases
    FlagDNN::flagdnn
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn)
  flagdnn_enable_warnings(flagdnn_test_ascend_matmul_contract)
  add_test(NAME core.ascend.matmul_validation_contract
    COMMAND flagdnn_test_ascend_matmul_contract)
  set_tests_properties(core.ascend.matmul_validation_contract PROPERTIES
    LABELS "core;ascend;no_device;validation;matmul"
    TIMEOUT 60)

  add_executable(flagdnn_test_ascend_convolution_fprop_contract
    benchmark/aclnn_convolution_provider.cpp
    convolution_fprop_contract_test.cpp
    functional/aclnn_convolution.cpp
    functional/aclnn_convolution_plan.cpp
    "${PROJECT_SOURCE_DIR}/tests/common/convolution.cpp")
  target_compile_features(
    flagdnn_test_ascend_convolution_fprop_contract PRIVATE cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_convolution_fprop_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_link_libraries(flagdnn_test_ascend_convolution_fprop_contract PRIVATE
    flagdnn_benchmark_cases
    FlagDNN::flagdnn
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn)
  flagdnn_enable_warnings(
    flagdnn_test_ascend_convolution_fprop_contract)
  add_test(NAME core.ascend.convolution_fprop_validation_contract
    COMMAND flagdnn_test_ascend_convolution_fprop_contract)
  set_tests_properties(
    core.ascend.convolution_fprop_validation_contract PROPERTIES
    LABELS "core;ascend;no_device;validation;convolution_fprop"
    TIMEOUT 60)

  add_library(flagdnn_test_ascend_convolution_fprop_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    functional/aclnn_convolution.cpp
    functional/aclnn_convolution_plan.cpp
    functional/convolution_fprop_runner.cpp)
  target_compile_features(
    flagdnn_test_ascend_convolution_fprop_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_convolution_fprop_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(
    flagdnn_test_ascend_convolution_fprop_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(
    flagdnn_test_ascend_convolution_fprop_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn)
  flagdnn_enable_warnings(
    flagdnn_test_ascend_convolution_fprop_adapter)

  flagdnn_register_functional_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_test_ascend_convolution_fprop_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS conv_fprop
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;device;convolution_fprop"
    TIMEOUT 2400
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  add_library(flagdnn_test_ascend_matmul_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    functional/aclnn_matmul.cpp
    functional/aclnn_matmul_plan.cpp
    functional/matmul_runner.cpp)
  target_compile_features(flagdnn_test_ascend_matmul_adapter PUBLIC
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_matmul_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(flagdnn_test_ascend_matmul_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(flagdnn_test_ascend_matmul_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn)
  flagdnn_enable_warnings(flagdnn_test_ascend_matmul_adapter)

  flagdnn_register_functional_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_test_ascend_matmul_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS matmul
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;device;matmul"
    TIMEOUT 1200
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  add_executable(flagdnn_test_ascend_development_runtime
    development_runtime_test.cpp)
  target_compile_features(flagdnn_test_ascend_development_runtime PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_ascend_development_runtime PRIVATE
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(flagdnn_test_ascend_development_runtime PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(flagdnn_test_ascend_development_runtime PRIVATE
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    Threads::Threads)
  flagdnn_enable_warnings(flagdnn_test_ascend_development_runtime)

  add_test(NAME core.ascend.development_runtime.resource_cache_snapshot_host
    COMMAND "${CMAKE_COMMAND}"
      "-DEXECUTABLE=$<TARGET_FILE:flagdnn_test_ascend_development_runtime>"
      "-DCOMPILER_EXECUTABLE=${FLAGDNN_CODEGEN_PYTHON}"
      "-DCOMPILER_ENTRY=${FLAGDNN_CODEGEN_COMPILER}"
      "-DBACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
      "-DMODE=resource_cache_snapshot_host"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/RunDevelopmentRuntime.cmake")
  set_tests_properties(
    core.ascend.development_runtime.resource_cache_snapshot_host PROPERTIES
    LABELS "core;ascend;no_device;resource;development"
    TIMEOUT 60)

  foreach(mode IN ITEMS
      ownership concurrency repeat workspace
      environment_drift no_active_sync no_active_sync_default)
    add_test(NAME integration.ascend.development_runtime.${mode}
      COMMAND flagdnn_test_ascend_development_runtime
        "${mode}"
        "${FLAGDNN_CODEGEN_PYTHON}"
        "${FLAGDNN_CODEGEN_COMPILER}")
    set_tests_properties(
      integration.ascend.development_runtime.${mode} PROPERTIES
      LABELS "integration;ascend;runtime;device;development"
      ENVIRONMENT
        "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
      RUN_SERIAL TRUE TIMEOUT 600)
  endforeach()

  add_test(NAME integration.ascend.development_runtime.cache
    COMMAND "${CMAKE_COMMAND}"
      "-DEXECUTABLE=$<TARGET_FILE:flagdnn_test_ascend_development_runtime>"
      "-DCOMPILER_EXECUTABLE=${FLAGDNN_CODEGEN_PYTHON}"
      "-DCOMPILER_ENTRY=${FLAGDNN_CODEGEN_COMPILER}"
      "-DBACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/RunDevelopmentRuntime.cmake")
  set_tests_properties(integration.ascend.development_runtime.cache PROPERTIES
    LABELS "integration;ascend;runtime;device;development;cache"
    RUN_SERIAL TRUE TIMEOUT 600)

  foreach(mode IN ITEMS
      resource_budget resource_graph_lifecycle resource_dual_handle)
    add_test(NAME integration.ascend.development_runtime.${mode}
      COMMAND "${CMAKE_COMMAND}"
        "-DEXECUTABLE=$<TARGET_FILE:flagdnn_test_ascend_development_runtime>"
        "-DCOMPILER_EXECUTABLE=${FLAGDNN_CODEGEN_PYTHON}"
        "-DCOMPILER_ENTRY=${FLAGDNN_CODEGEN_COMPILER}"
        "-DBACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
        "-DMODE=${mode}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/RunDevelopmentRuntime.cmake")
    set_tests_properties(
      integration.ascend.development_runtime.${mode} PROPERTIES
      LABELS "integration;ascend;runtime;device;development;cache;resource"
      RUN_SERIAL TRUE TIMEOUT 1200)
  endforeach()

  add_library(flagdnn_test_ascend_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    aclnn_unary_runtime.cpp
    functional/aclnn_add.cpp
    functional/aclnn_binary_pointwise.cpp
    functional/aclnn_ternary_pointwise.cpp
    functional/aclnn_unary_pointwise.cpp
    functional/add_runner.cpp
    functional/pointwise_runner.cpp)
  target_compile_features(flagdnn_test_ascend_adapter PUBLIC cxx_std_20)
  target_include_directories(flagdnn_test_ascend_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(flagdnn_test_ascend_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(flagdnn_test_ascend_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn)
  flagdnn_enable_warnings(flagdnn_test_ascend_adapter)

  add_library(flagdnn_test_ascend_nn_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    aclnn_unary_runtime.cpp
    functional/aclnn_add.cpp
    functional/aclnn_binary_pointwise.cpp
    functional/aclnn_ternary_pointwise.cpp
    functional/aclnn_unary_pointwise.cpp
    functional/add_runner.cpp
    functional/pointwise_runner.cpp)
  target_compile_features(flagdnn_test_ascend_nn_adapter PUBLIC cxx_std_20)
  target_include_directories(flagdnn_test_ascend_nn_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(flagdnn_test_ascend_nn_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN=1
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(flagdnn_test_ascend_nn_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn)
  flagdnn_enable_warnings(flagdnn_test_ascend_nn_adapter)

  flagdnn_register_functional_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_test_ascend_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS
      add sub mul scale div mod pow min max
      binary_select
      cmp_eq cmp_neq cmp_gt cmp_ge cmp_lt cmp_le
      logical_not logical_and logical_or
      identity neg abs relu leaky_relu ceil floor reciprocal sqrt rsqrt
      exp log tanh sin cos tan erf
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;device"
    TIMEOUT 1200
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  flagdnn_register_functional_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_test_ascend_reduction_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS reduction
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;device;reduction"
    TIMEOUT 1200
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  # Layout references use CANN's precise math primitives and own a dedicated
  # runner so pointwise symbol resolution and case-count contracts stay
  # isolated.
  add_library(flagdnn_test_ascend_layout_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    functional/aclnn_layout.cpp
    functional/layout_runner.cpp)
  target_compile_features(
    flagdnn_test_ascend_layout_adapter PUBLIC cxx_std_20)
  target_include_directories(flagdnn_test_ascend_layout_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(flagdnn_test_ascend_layout_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(flagdnn_test_ascend_layout_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn)
  flagdnn_enable_warnings(flagdnn_test_ascend_layout_adapter)

  flagdnn_register_functional_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_test_ascend_layout_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS reshape transpose slice
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;device;layout"
    TIMEOUT 1200
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  flagdnn_register_functional_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_test_ascend_nn_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS sigmoid sigmoid_backward elu gelu softplus swish gelu_approx_tanh
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;device"
    TIMEOUT 1200
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  add_executable(flagdnn_test_ascend_conv_bias_relu_contract
    conv_bias_relu_contract_test.cpp
    functional/aclnn_conv_bias_relu.cpp
    functional/aclnn_convolution.cpp
    functional/aclnn_convolution_plan.cpp)
  target_compile_features(
    flagdnn_test_ascend_conv_bias_relu_contract PRIVATE cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_conv_bias_relu_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_link_libraries(
    flagdnn_test_ascend_conv_bias_relu_contract PRIVATE
    flagdnn_test_ascend_nn_adapter)
  flagdnn_enable_warnings(
    flagdnn_test_ascend_conv_bias_relu_contract)
  add_test(NAME core.ascend.conv_bias_relu_validation_contract
    COMMAND flagdnn_test_ascend_conv_bias_relu_contract)
  set_tests_properties(
    core.ascend.conv_bias_relu_validation_contract PROPERTIES
    LABELS "core;ascend;no_device;validation;conv_bias_relu"
    TIMEOUT 60)

  add_library(flagdnn_test_ascend_conv_bias_relu_adapter STATIC
    functional/aclnn_conv_bias_relu.cpp
    functional/aclnn_convolution.cpp
    functional/aclnn_convolution_plan.cpp
    functional/conv_bias_relu_runner.cpp)
  target_compile_features(
    flagdnn_test_ascend_conv_bias_relu_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_test_ascend_conv_bias_relu_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(
    flagdnn_test_ascend_conv_bias_relu_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(
    flagdnn_test_ascend_conv_bias_relu_adapter PUBLIC
    flagdnn_test_ascend_nn_adapter)
  flagdnn_enable_warnings(flagdnn_test_ascend_conv_bias_relu_adapter)

  flagdnn_register_functional_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_test_ascend_conv_bias_relu_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS conv_bias_relu
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;device;conv_bias_relu"
    TIMEOUT 2400
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  # AddSquare uses the same Ascend binary kernels, but its public test entry
  # resolves a composite-specific runner and an independent ACLNN Mul -> Add
  # reference.  Keep it in a separate archive so the generic pointwise runner
  # remains the only run_pointwise_functional_test implementation.
  add_library(flagdnn_test_ascend_add_square_adapter STATIC
    functional/aclnn_add_square.cpp
    functional/add_square_runner.cpp)
  target_compile_features(
    flagdnn_test_ascend_add_square_adapter PUBLIC cxx_std_20)
  target_include_directories(flagdnn_test_ascend_add_square_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend")
  target_compile_definitions(flagdnn_test_ascend_add_square_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
  target_link_libraries(flagdnn_test_ascend_add_square_adapter PUBLIC
    flagdnn_test_ascend_adapter)
  flagdnn_enable_warnings(flagdnn_test_ascend_add_square_adapter)

  flagdnn_register_functional_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_test_ascend_add_square_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS add_square
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;device;graph"
    TIMEOUT 1200
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  add_test(NAME integration.ascend.dependency_boundary
    COMMAND "${CMAKE_COMMAND}"
      "-DCORE_LIBRARY=$<TARGET_FILE:flagdnn>"
      "-DBACKEND_LIBRARY=$<TARGET_FILE:flagdnn_backend_ascend>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/VerifyNativeDependencies.cmake")
  set_tests_properties(integration.ascend.dependency_boundary PROPERTIES
    LABELS "integration;ascend;dependency")

  foreach(_flagdnn_validation_operator IN ITEMS
      add sub mul scale div mod pow min max add_square
      binary_select
      cmp_eq cmp_neq cmp_gt cmp_ge cmp_lt cmp_le
      logical_not logical_and logical_or
      identity neg abs relu leaky_relu ceil floor reciprocal sqrt rsqrt
      exp log tanh sin cos tan erf
      reduction
      matmul
      conv_fprop conv_bias_relu
      reshape transpose slice
      sigmoid sigmoid_backward elu gelu softplus swish gelu_approx_tanh
      batchnorm batchnorm_inference rmsnorm layernorm)
    if(_flagdnn_validation_operator MATCHES
        "^(sigmoid|sigmoid_backward|elu|gelu|softplus|swish|gelu_approx_tanh|batchnorm|batchnorm_inference|rmsnorm|layernorm|matmul|conv_fprop|conv_bias_relu)$")
      set(_flagdnn_validation_requires_opapi_nn TRUE)
    else()
      set(_flagdnn_validation_requires_opapi_nn FALSE)
    endif()
    add_test(
      NAME integration.ascend.reference_dependency_boundary.${_flagdnn_validation_operator}
      COMMAND "${CMAKE_COMMAND}"
        "-DREFERENCE_EXECUTABLE=$<TARGET_FILE:flagdnn_test_ascend_${_flagdnn_validation_operator}>"
        "-DREFERENCE_REQUIRES_OPAPI_NN=${_flagdnn_validation_requires_opapi_nn}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/VerifyReferenceDependencies.cmake")
    set_tests_properties(
      integration.ascend.reference_dependency_boundary.${_flagdnn_validation_operator}
      PROPERTIES LABELS "integration;ascend;aclnn;dependency")
  endforeach()
endif()

if(FLAGDNN_BUILD_BENCHMARKS)
  add_library(flagdnn_benchmark_ascend_layernorm_adapter STATIC
    layernorm_validation.cpp
    functional/aclnn_layernorm.cpp
    functional/aclnn_layernorm_plan.cpp
    benchmark/aclnn_layernorm_provider.cpp
    benchmark/layernorm_runner.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
  target_compile_features(
    flagdnn_benchmark_ascend_layernorm_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_benchmark_ascend_layernorm_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend"
    "${PROJECT_SOURCE_DIR}/src")
  target_compile_definitions(
    flagdnn_benchmark_ascend_layernorm_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
    FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
    FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
    FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
    FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
    FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256="${_flagdnn_validation_opapi_nn_sha256}")
  target_link_libraries(
    flagdnn_benchmark_ascend_layernorm_adapter PUBLIC
    flagdnn_benchmark_cases
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn
    ${CMAKE_DL_LIBS})
  flagdnn_enable_warnings(flagdnn_benchmark_ascend_layernorm_adapter)

  flagdnn_register_benchmark_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_benchmark_ascend_layernorm_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS layernorm
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;raw_event;device;normalization"
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  add_library(flagdnn_benchmark_ascend_rmsnorm_adapter STATIC
    rmsnorm_validation.cpp
    functional/aclnn_rmsnorm.cpp
    functional/aclnn_rmsnorm_plan.cpp
    benchmark/aclnn_rmsnorm_provider.cpp
    benchmark/rmsnorm_runner.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
  target_compile_features(
    flagdnn_benchmark_ascend_rmsnorm_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_benchmark_ascend_rmsnorm_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend"
    "${PROJECT_SOURCE_DIR}/src")
  target_compile_definitions(
    flagdnn_benchmark_ascend_rmsnorm_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
    FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
    FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
    FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
    FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
    FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256="${_flagdnn_validation_opapi_nn_sha256}")
  target_link_libraries(
    flagdnn_benchmark_ascend_rmsnorm_adapter PUBLIC
    flagdnn_benchmark_cases
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn
    ${CMAKE_DL_LIBS})
  flagdnn_enable_warnings(flagdnn_benchmark_ascend_rmsnorm_adapter)

  flagdnn_register_benchmark_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_benchmark_ascend_rmsnorm_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS rmsnorm
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;raw_event;device;normalization"
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  add_library(flagdnn_benchmark_ascend_batchnorm_inference_adapter STATIC
    batchnorm_inference_validation.cpp
    functional/aclnn_batchnorm_inference.cpp
    benchmark/aclnn_batchnorm_inference_provider.cpp
    benchmark/batchnorm_inference_runner.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
  target_compile_features(
    flagdnn_benchmark_ascend_batchnorm_inference_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_benchmark_ascend_batchnorm_inference_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend"
    "${PROJECT_SOURCE_DIR}/src")
  target_compile_definitions(
    flagdnn_benchmark_ascend_batchnorm_inference_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
    FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
    FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
    FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
    FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
    FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256="${_flagdnn_validation_opapi_nn_sha256}")
  target_link_libraries(
    flagdnn_benchmark_ascend_batchnorm_inference_adapter PUBLIC
    flagdnn_benchmark_cases
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn
    ${CMAKE_DL_LIBS})
  flagdnn_enable_warnings(
    flagdnn_benchmark_ascend_batchnorm_inference_adapter)

  flagdnn_register_benchmark_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_benchmark_ascend_batchnorm_inference_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS batchnorm_inference
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;raw_event;device;normalization"
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  add_library(flagdnn_benchmark_ascend_batchnorm_adapter STATIC
    batchnorm_validation.cpp
    functional/aclnn_batchnorm.cpp
    benchmark/aclnn_batchnorm_provider.cpp
    benchmark/batchnorm_runner.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
  target_compile_features(
    flagdnn_benchmark_ascend_batchnorm_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_benchmark_ascend_batchnorm_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend"
    "${PROJECT_SOURCE_DIR}/src")
  target_compile_definitions(
    flagdnn_benchmark_ascend_batchnorm_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
    FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
    FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
    FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
    FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
    FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256="${_flagdnn_validation_opapi_nn_sha256}")
  target_link_libraries(
    flagdnn_benchmark_ascend_batchnorm_adapter PUBLIC
    flagdnn_benchmark_cases
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn
    ${CMAKE_DL_LIBS})
  flagdnn_enable_warnings(flagdnn_benchmark_ascend_batchnorm_adapter)

  flagdnn_register_benchmark_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_benchmark_ascend_batchnorm_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS batchnorm
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;raw_event;device;normalization"
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  add_library(flagdnn_benchmark_ascend_adapter STATIC
    aclnn_unary_runtime.cpp
    "${PROJECT_SOURCE_DIR}/tests/common/reduction.cpp"
    functional/aclnn_reduction.cpp
    functional/aclnn_reduction_plan.cpp
    reduction_validation.cpp
    benchmark/aclnn_provider.cpp
    benchmark/aclnn_unary_provider.cpp
    benchmark/aclnn_reduction_plan.cpp
    benchmark/aclnn_reduction_provider.cpp
    benchmark/runner.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
  target_compile_features(flagdnn_benchmark_ascend_adapter PUBLIC cxx_std_20)
  target_include_directories(flagdnn_benchmark_ascend_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend"
    "${PROJECT_SOURCE_DIR}/src")
  target_compile_definitions(flagdnn_benchmark_ascend_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
    FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
    FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
    FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
    FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
    FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}")
  target_link_libraries(flagdnn_benchmark_ascend_adapter PUBLIC
    flagdnn_benchmark_cases
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn
    ${CMAKE_DL_LIBS})
  flagdnn_enable_warnings(flagdnn_benchmark_ascend_adapter)

  add_executable(flagdnn_test_ascend_benchmark_provider_contract
    benchmark_provider_contract_test.cpp)
  target_compile_features(
    flagdnn_test_ascend_benchmark_provider_contract PRIVATE cxx_std_20)
  target_link_libraries(
    flagdnn_test_ascend_benchmark_provider_contract PRIVATE
    flagdnn_benchmark_ascend_adapter)
  flagdnn_enable_warnings(
    flagdnn_test_ascend_benchmark_provider_contract)
  add_test(
    NAME core.ascend.benchmark_provider_contract
    COMMAND flagdnn_test_ascend_benchmark_provider_contract)
  set_tests_properties(
    core.ascend.benchmark_provider_contract
    PROPERTIES LABELS "core;ascend;benchmark;no_device")

  add_library(flagdnn_benchmark_ascend_nn_adapter STATIC
    aclnn_unary_runtime.cpp
    "${PROJECT_SOURCE_DIR}/tests/common/convolution.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/matmul.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/reduction.cpp"
    functional/aclnn_convolution.cpp
    functional/aclnn_convolution_plan.cpp
    functional/aclnn_matmul.cpp
    functional/aclnn_matmul_plan.cpp
    functional/aclnn_reduction.cpp
    functional/aclnn_reduction_plan.cpp
    reduction_validation.cpp
    benchmark/aclnn_convolution_provider.cpp
    benchmark/aclnn_provider.cpp
    benchmark/aclnn_matmul_provider.cpp
    benchmark/aclnn_unary_provider.cpp
    benchmark/aclnn_reduction_plan.cpp
    benchmark/aclnn_reduction_provider.cpp
    benchmark/runner.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
  target_compile_features(
    flagdnn_benchmark_ascend_nn_adapter PUBLIC cxx_std_20)
  target_include_directories(flagdnn_benchmark_ascend_nn_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend"
    "${PROJECT_SOURCE_DIR}/src")
  target_compile_definitions(flagdnn_benchmark_ascend_nn_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP=1
    FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL=1
    FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN=1
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
    FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
    FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
    FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
    FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
    FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256="${_flagdnn_validation_opapi_nn_sha256}")
  target_link_libraries(flagdnn_benchmark_ascend_nn_adapter PUBLIC
    flagdnn_benchmark_cases
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn_nn
    ${CMAKE_DL_LIBS})
  flagdnn_enable_warnings(flagdnn_benchmark_ascend_nn_adapter)

  add_library(flagdnn_benchmark_ascend_reduction_adapter STATIC
    aclnn_unary_runtime.cpp
    "${PROJECT_SOURCE_DIR}/tests/common/reduction.cpp"
    functional/aclnn_reduction.cpp
    functional/aclnn_reduction_plan.cpp
    reduction_validation.cpp
    benchmark/aclnn_provider.cpp
    benchmark/aclnn_unary_provider.cpp
    benchmark/aclnn_reduction_plan.cpp
    benchmark/aclnn_reduction_provider.cpp
    benchmark/runner.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
  target_compile_features(flagdnn_benchmark_ascend_reduction_adapter PUBLIC
    cxx_std_20)
  target_include_directories(flagdnn_benchmark_ascend_reduction_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend"
    "${PROJECT_SOURCE_DIR}/src")
  target_compile_definitions(flagdnn_benchmark_ascend_reduction_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
    FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
    FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
    FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
    FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
    FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}")
  target_link_libraries(flagdnn_benchmark_ascend_reduction_adapter PUBLIC
    flagdnn_benchmark_cases
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn
    ${CMAKE_DL_LIBS})
  flagdnn_enable_warnings(flagdnn_benchmark_ascend_reduction_adapter)

  flagdnn_register_benchmark_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_benchmark_ascend_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS
      add sub mul scale div mod pow min max
      binary_select
      cmp_eq cmp_neq cmp_gt cmp_ge cmp_lt cmp_le
      logical_not logical_and logical_or
      identity neg abs relu leaky_relu ceil floor reciprocal sqrt rsqrt
      exp log tanh sin cos tan erf
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;raw_event;device"
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  flagdnn_register_benchmark_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_benchmark_ascend_reduction_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS reduction
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;raw_event;device;reduction"
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  # Layout benchmarks compare directly with the transactional ACLNN layout
  # executable without importing the pointwise runner entry point.
  add_library(flagdnn_benchmark_ascend_layout_adapter STATIC
    "${PROJECT_SOURCE_DIR}/tests/common/layout.cpp"
    functional/aclnn_layout.cpp
    benchmark/aclnn_layout_provider.cpp
    benchmark/layout_runner.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
  target_compile_features(
    flagdnn_benchmark_ascend_layout_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_benchmark_ascend_layout_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/backends/ascend"
    "${PROJECT_SOURCE_DIR}/src")
  target_compile_definitions(
    flagdnn_benchmark_ascend_layout_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
    FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
    FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
    FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
    FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
    FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}")
  target_link_libraries(flagdnn_benchmark_ascend_layout_adapter PUBLIC
    flagdnn_benchmark_cases
    FlagDNN::flagdnn
    flagdnn_validation_ascend_platform
    flagdnn_validation_ascend_tensor_io
    flagdnn_validation_ascend_aclnn
    ${CMAKE_DL_LIBS})
  flagdnn_enable_warnings(flagdnn_benchmark_ascend_layout_adapter)

  flagdnn_register_benchmark_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_benchmark_ascend_layout_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS reshape transpose slice
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;raw_event;device;layout"
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  flagdnn_register_benchmark_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_benchmark_ascend_nn_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS sigmoid sigmoid_backward elu gelu softplus swish gelu_approx_tanh
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;raw_event;device"
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  flagdnn_register_benchmark_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_benchmark_ascend_nn_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS matmul
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;raw_event;device;matmul"
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  flagdnn_register_benchmark_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_benchmark_ascend_nn_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS conv_fprop
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;raw_event;device;convolution_fprop"
    TIMEOUT 7200
    BUILD_RPATH "${_flagdnn_validation_library_directory}")

  # The composite benchmark owns a distinct run_benchmark_suite entry point;
  # isolate it from the binary pointwise adapter to keep symbol resolution
  # deterministic while reusing the same platform/reference primitives.
  add_library(flagdnn_benchmark_ascend_add_square_adapter STATIC
    benchmark/aclnn_add_square_provider.cpp
    benchmark/add_square_runner.cpp)
  target_compile_features(
    flagdnn_benchmark_ascend_add_square_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_benchmark_ascend_add_square_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/backends/ascend"
    "${PROJECT_SOURCE_DIR}/src")
  target_compile_definitions(
    flagdnn_benchmark_ascend_add_square_adapter PRIVATE
    FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
    FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
    FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
    FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
    FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
    FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
    FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
    FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}")
  target_link_libraries(flagdnn_benchmark_ascend_add_square_adapter PUBLIC
    flagdnn_benchmark_ascend_adapter)
  flagdnn_enable_warnings(flagdnn_benchmark_ascend_add_square_adapter)

  flagdnn_register_benchmark_suite(
    PLATFORM ascend
    ADAPTER_TARGET flagdnn_benchmark_ascend_add_square_adapter
    BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS add_square
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;exact_reference;raw_event;device;graph"
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_validation_library_directory}")
endif()
