cmake_minimum_required(VERSION 3.23)

foreach(required IN ITEMS
    PROJECT_BUILD_DIR SOURCE_ROOT TEST_ROOT CANN_ROOT CANN_VERSION CODEGEN_PYTHON
    LTJ_LIBRARY LTJ_STANDALONE LTJ_CONFIG LTJ_INCLUDE RESOURCE_INSTALL_DIR
    LIB_INSTALL_DIR PYTHON_MODULE_ROOT)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "VerifyInstalledAscendConsumer requires ${required}")
  endif()
endforeach()

if(NOT DEFINED RUN_ASCEND_DEVICE_TEST)
  set(RUN_ASCEND_DEVICE_TEST OFF)
endif()
if(RUN_ASCEND_DEVICE_TEST)
  foreach(required IN ITEMS PYTHON_LIBRARY ASCEND_DRIVER_ROOT)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
      message(FATAL_ERROR
        "RUN_ASCEND_DEVICE_TEST requires ${required}")
    endif()
  endforeach()
  if(NOT IS_ABSOLUTE "${TEST_ROOT}")
    message(FATAL_ERROR
      "RUN_ASCEND_DEVICE_TEST requires an absolute TEST_ROOT")
  endif()
endif()

file(READ
  "${SOURCE_ROOT}/tests/core/installed_consumer/ascend_add.cpp"
  installed_ascend_consumer_source)
foreach(required_fragment IN ITEMS
    "graph.matmul("
    "matmul_a"
    "matmul_b"
    "matmul_output"
    "installed Ascend MatMul output differs"
    "installed Ascend MatMul graph modified an input")
  string(FIND "${installed_ascend_consumer_source}"
    "${required_fragment}" required_fragment_offset)
  if(required_fragment_offset EQUAL -1)
    message(FATAL_ERROR
      "Installed Ascend consumer is missing MatMul contract fragment: "
      "${required_fragment}")
  endif()
endforeach()

foreach(required_fragment IN ITEMS
    "graph.conv_fprop("
    "convolution_input"
    "convolution_filter"
    "convolution_output"
    "installed Ascend ConvFprop output differs"
    "installed Ascend ConvFprop graph modified an input")
  string(FIND "${installed_ascend_consumer_source}"
    "${required_fragment}" required_fragment_offset)
  if(required_fragment_offset EQUAL -1)
    message(FATAL_ERROR
      "Installed Ascend consumer is missing ConvFprop contract fragment: "
      "${required_fragment}")
  endif()
endforeach()

foreach(required_fragment IN ITEMS
    "graph.rmsnorm("
    "rmsnorm_input"
    "rmsnorm_output"
    "rmsnorm_inv_variance"
    "installed Ascend RMSNorm/ReLU output differs"
    "installed Ascend RMSNorm inverse variance differs"
    "installed Ascend RMSNorm graph modified an input")
  string(FIND "${installed_ascend_consumer_source}"
    "${required_fragment}" required_fragment_offset)
  if(required_fragment_offset EQUAL -1)
    message(FATAL_ERROR
      "Installed Ascend consumer is missing RMSNorm contract fragment: "
      "${required_fragment}")
  endif()
endforeach()

foreach(required_fragment IN ITEMS
    "graph.layernorm("
    "layernorm_input"
    "layernorm_output"
    "layernorm_mean"
    "layernorm_inv_variance"
    "installed Ascend LayerNorm/ReLU output differs"
    "installed Ascend LayerNorm mean differs"
    "installed Ascend LayerNorm inverse variance differs"
    "installed Ascend LayerNorm graph modified an input")
  string(FIND "${installed_ascend_consumer_source}"
    "${required_fragment}" required_fragment_offset)
  if(required_fragment_offset EQUAL -1)
    message(FATAL_ERROR
      "Installed Ascend consumer is missing LayerNorm contract fragment: "
      "${required_fragment}")
  endif()
endforeach()

foreach(required_fragment IN ITEMS
    "graph.batchnorm("
    "batchnorm_training_input"
    "batchnorm_training_output"
    "batchnorm_training_mean"
    "batchnorm_training_inv_variance"
    "batchnorm_training_next_running_mean"
    "batchnorm_training_next_running_variance"
    "installed Ascend BatchNorm training output differs"
    "installed Ascend BatchNorm training statistic differs"
    "installed Ascend BatchNorm training graph modified an input")
  string(FIND "${installed_ascend_consumer_source}"
    "${required_fragment}" required_fragment_offset)
  if(required_fragment_offset EQUAL -1)
    message(FATAL_ERROR
      "Installed Ascend consumer is missing BatchNorm training contract "
      "fragment: ${required_fragment}")
  endif()
endforeach()

set(prefix "${TEST_ROOT}/prefix")
set(consumer_build "${TEST_ROOT}/consumer")
file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")

set(install_command
  "${CMAKE_COMMAND}" --install "${PROJECT_BUILD_DIR}" --prefix "${prefix}")
if(DEFINED BUILD_CONFIG AND NOT BUILD_CONFIG STREQUAL "")
  list(APPEND install_command --config "${BUILD_CONFIG}")
endif()
execute_process(
  COMMAND ${install_command}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "FlagDNN temporary install failed:\n${output}\n${error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${SOURCE_ROOT}/tests/core/installed_consumer"
    -B "${consumer_build}"
    "-DCMAKE_PREFIX_PATH=${prefix}"
    -DFLAGDNN_INSTALLED_ASCEND_ADD=ON
    "-DCANN_ROOT=${CANN_ROOT}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "Installed Ascend consumer configure failed:\n${output}\n${error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${consumer_build}"
    --target
      flagdnn_installed_c_consumer
      flagdnn_installed_cpp_consumer
      flagdnn_installed_ascend_add
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "Installed Ascend consumer build failed:\n${output}\n${error}")
endif()

# The Stage-0 contract runs only the device-neutral C and C++ consumers.  The
# Ascend executable is deliberately only configured and linked here.
execute_process(
  COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build}"
    --output-on-failure -R "^installed\\.(c|cpp)$"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "Installed C/C++ consumer smoke failed:\n${output}\n${error}")
endif()

set(resources
  "backends/ascend/compiler.py"
  "backends/ascend/compiler_identity.py"
  "backends/ascend/add_plan.py"
  "backends/ascend/tuning_decoder.py"
  "backends/ascend/capabilities.json"
  "backends/ascend/sandbox_policy.json"
  "backends/ascend/workspace_attestation.py"
  "backends/ascend/kernels/registry.json"
  "backends/ascend/kernels/binary.py"
  "backends/ascend/kernels/convolution.py"
  "backends/ascend/kernels/layout.py"
  "backends/ascend/kernels/matmul.py"
  "backends/ascend/kernels/normalization.py"
  "backends/ascend/kernels/reduction.py"
  "backends/ascend/kernels/ternary.py"
  "backends/ascend/kernels/unary.py"
  "backends/ascend/tuning/common.yaml"
  "compiler/flagdnn_codegen/main.py"
  "compiler/flagdnn_codegen/provider_loader.py"
  "compiler/flagdnn_codegen/kernel_registry.py"
  "kernels/common/binary.py"
  "kernels/common/layout.py"
  "kernels/common/reduction.py"
  "kernels/common/ternary.py"
  "kernels/common/unary.py")
foreach(relative IN LISTS resources)
  set(source "${SOURCE_ROOT}/${relative}")
  set(installed "${prefix}/${RESOURCE_INSTALL_DIR}/${relative}")
  if(NOT EXISTS "${source}" OR NOT EXISTS "${installed}")
    message(FATAL_ERROR "Installed compiler resource is missing: ${relative}")
  endif()
  file(SHA256 "${source}" source_sha256)
  file(SHA256 "${installed}" installed_sha256)
  if(NOT source_sha256 STREQUAL installed_sha256)
    message(FATAL_ERROR "Installed compiler resource differs: ${relative}")
  endif()
endforeach()

set(identity_output "${TEST_ROOT}/installed-compiler.identity")
set(identity_environment
  PYTHONDONTWRITEBYTECODE=1
  PYTHONNOUSERSITE=1
  PYTHONHASHSEED=0
  PYTHONSAFEPATH=1
  TORCH_DEVICE_BACKEND_AUTOLOAD=0
  TRITON_JIT_BACKEND=NPU
  TRITON_BACKEND=torch_npu
  FLAGDNN_ASCEND_REQUIRE_COMPILER_MODULES=1
  "PYTHONPATH=${PYTHON_MODULE_ROOT}"
  "ASCEND_HOME_PATH=${CANN_ROOT}"
  "FLAGDNN_LIBTRITON_JIT_LIBRARY=${LTJ_LIBRARY}"
  "FLAGDNN_TRITON_JIT_STANDALONE_COMPILER=${LTJ_STANDALONE}"
  "FLAGDNN_TRITON_JIT_CONFIG=${LTJ_CONFIG}"
  "FLAGDNN_TRITON_JIT_INCLUDE_DIRECTORY=${LTJ_INCLUDE}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    ${identity_environment}
    "${CODEGEN_PYTHON}"
    "${prefix}/${RESOURCE_INSTALL_DIR}/compiler/flagdnn_codegen/main.py"
    --identify
    --backend ascend
    "--target=ascend_Ascend910B1_cann_${CANN_VERSION}_aic_24"
    --execution-engine libtriton_jit
    --identity-output "${identity_output}"
    --quiet
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
  TIMEOUT 30)
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "Installed Ascend compiler discovery failed:\n${output}\n${error}")
endif()
file(STRINGS "${identity_output}" identity_lines LIMIT_COUNT 1)
list(LENGTH identity_lines identity_line_count)
if(NOT identity_line_count EQUAL 1)
  message(FATAL_ERROR "Installed Ascend compiler identity is missing")
endif()
list(GET identity_lines 0 identity)
string(LENGTH "${identity}" identity_length)
if(NOT identity_length EQUAL 64 OR NOT identity MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "Installed Ascend compiler identity is invalid")
endif()

set(source_identity_output "${TEST_ROOT}/source-compiler.identity")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    ${identity_environment}
    "${CODEGEN_PYTHON}"
    "${SOURCE_ROOT}/compiler/flagdnn_codegen/main.py"
    --identify
    --backend ascend
    "--target=ascend_Ascend910B1_cann_${CANN_VERSION}_aic_24"
    --execution-engine libtriton_jit
    --identity-output "${source_identity_output}"
    --quiet
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
  TIMEOUT 30)
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "Source Ascend compiler discovery failed:\n${output}\n${error}")
endif()
file(STRINGS "${source_identity_output}" source_identity_lines LIMIT_COUNT 1)
list(LENGTH source_identity_lines source_identity_line_count)
if(NOT source_identity_line_count EQUAL 1)
  message(FATAL_ERROR "Source Ascend compiler identity is missing")
endif()
list(GET source_identity_lines 0 source_identity)
if(NOT source_identity STREQUAL identity)
  message(FATAL_ERROR
    "Source and installed Ascend compiler identities differ")
endif()

find_program(READELF_EXECUTABLE readelf REQUIRED)
set(installed_plugin
  "${prefix}/${LIB_INSTALL_DIR}/libflagdnn_backend_ascend.so")
execute_process(
  COMMAND "${READELF_EXECUTABLE}" -d "${installed_plugin}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Cannot inspect installed Ascend plugin: ${error}")
endif()
string(REPLACE "\n" ";" dynamic_lines "${dynamic_section}")
set(path_lines)
foreach(line IN LISTS dynamic_lines)
  if(line MATCHES "\\((RPATH|RUNPATH)\\)")
    string(STRIP "${line}" line)
    list(APPEND path_lines "${line}")
  endif()
endforeach()
list(LENGTH path_lines path_line_count)
if(NOT path_line_count EQUAL 1)
  message(FATAL_ERROR
    "Installed Ascend plugin must contain exactly one RUNPATH entry:\n${dynamic_section}")
endif()
list(GET path_lines 0 path_line)
if(NOT path_line MATCHES "\\(RUNPATH\\).*Library runpath: \\[\\$ORIGIN\\]$")
  message(FATAL_ERROR
    "Installed Ascend plugin RUNPATH must be exactly $ORIGIN:\n${path_line}")
endif()

# Stage 0 deliberately stops after the device-neutral consumers have run and
# the Ascend consumer has configured and linked.  A device runner may opt in
# to the default trusted-local smoke without changing the no-device contract
# above.
if(NOT RUN_ASCEND_DEVICE_TEST)
  return()
endif()

file(REAL_PATH "${TEST_ROOT}" canonical_test_root)
file(REAL_PATH "${CANN_ROOT}" canonical_cann_root)
file(REAL_PATH "${PYTHON_MODULE_ROOT}" canonical_python_module_root)
file(REAL_PATH "${LTJ_LIBRARY}" canonical_ltj_library)
file(REAL_PATH "${PYTHON_LIBRARY}" canonical_python_library)
file(REAL_PATH "${ASCEND_DRIVER_ROOT}" canonical_driver_root)
foreach(required_directory IN ITEMS
    canonical_test_root canonical_cann_root canonical_python_module_root
    canonical_driver_root)
  if(NOT IS_DIRECTORY "${${required_directory}}")
    message(FATAL_ERROR
      "Installed Ascend device smoke directory is missing: "
      "${${required_directory}}")
  endif()
endforeach()
foreach(required_file IN ITEMS canonical_ltj_library canonical_python_library)
  if(NOT EXISTS "${${required_file}}" OR
     IS_DIRECTORY "${${required_file}}")
    message(FATAL_ERROR
      "Installed Ascend device smoke file is missing: ${${required_file}}")
  endif()
endforeach()

set(prefix_library_directory "${prefix}/${LIB_INSTALL_DIR}")
file(REAL_PATH "${prefix_library_directory}" prefix_library_directory)
set(installed_core "${prefix_library_directory}/libflagdnn.so")
set(installed_plugin
  "${prefix_library_directory}/libflagdnn_backend_ascend.so")
foreach(installed_library IN ITEMS installed_core installed_plugin)
  if(NOT EXISTS "${${installed_library}}" OR
     IS_DIRECTORY "${${installed_library}}")
    message(FATAL_ERROR
      "Installed Ascend device smoke library is missing: "
      "${${installed_library}}")
  endif()
  file(REAL_PATH "${${installed_library}}" installed_library_real)
  get_filename_component(installed_library_directory
    "${installed_library_real}" DIRECTORY)
  if(NOT installed_library_directory STREQUAL prefix_library_directory)
    message(FATAL_ERROR
      "Installed Ascend device smoke library escaped the temporary prefix: "
      "${installed_library_real}")
  endif()
endforeach()

set(consumer_executable
  "${consumer_build}/flagdnn_installed_ascend_add")
if(DEFINED BUILD_CONFIG AND NOT BUILD_CONFIG STREQUAL "" AND
   EXISTS "${consumer_build}/${BUILD_CONFIG}/flagdnn_installed_ascend_add")
  set(consumer_executable
    "${consumer_build}/${BUILD_CONFIG}/flagdnn_installed_ascend_add")
endif()
if(NOT EXISTS "${consumer_executable}" OR IS_DIRECTORY "${consumer_executable}")
  message(FATAL_ERROR
    "Installed Ascend consumer executable is missing: ${consumer_executable}")
endif()

# The executable must resolve the public SDK from the temporary install, not
# from the source build or a system-wide FlagDNN installation.  The exact
# LD_LIBRARY_PATH assembled below also starts with this directory.
execute_process(
  COMMAND "${READELF_EXECUTABLE}" -d "${consumer_executable}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE consumer_dynamic_section
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "Cannot inspect installed Ascend consumer: ${error}")
endif()
if(NOT consumer_dynamic_section MATCHES
   "Shared library: \\[libflagdnn\\.so(\\.[0-9]+)?\\]")
  message(FATAL_ERROR
    "Installed Ascend consumer does not depend on the FlagDNN SDK:\n"
    "${consumer_dynamic_section}")
endif()
string(REPLACE "\n" ";" consumer_dynamic_lines
  "${consumer_dynamic_section}")
set(consumer_path_lines)
foreach(line IN LISTS consumer_dynamic_lines)
  if(line MATCHES "\\((RPATH|RUNPATH)\\)")
    string(STRIP "${line}" line)
    list(APPEND consumer_path_lines "${line}")
  endif()
endforeach()
list(LENGTH consumer_path_lines consumer_path_line_count)
if(NOT consumer_path_line_count EQUAL 1)
  message(FATAL_ERROR
    "Installed Ascend consumer must contain exactly one RUNPATH entry:\n"
    "${consumer_dynamic_section}")
endif()
list(GET consumer_path_lines 0 consumer_path_line)
string(REGEX MATCH "Library runpath: \\[([^]]*)\\]"
  unused "${consumer_path_line}")
set(consumer_runpath "${CMAKE_MATCH_1}")
string(REPLACE ":" ";" consumer_runpath_entries "${consumer_runpath}")
list(FIND consumer_runpath_entries "${prefix_library_directory}"
  prefix_runpath_index)
if(prefix_runpath_index EQUAL -1)
  message(FATAL_ERROR
    "Installed Ascend consumer RUNPATH does not name the temporary prefix: "
    "${consumer_path_line}")
endif()

# Resolve Torch without importing it: importing torch_npu before FlagDNN owns
# and freezes the embedded interpreter is outside the supported contract.
execute_process(
  COMMAND "${CODEGEN_PYTHON}" -I -c
    "import importlib.util,pathlib,sys; s=importlib.util.find_spec('torch'); r=pathlib.Path(s.origin).resolve().parent if s and s.origin else None; d=[] if r is None else [r/'lib',r.parent/'torch.libs']; p=[str(x.resolve()) for x in d if x.is_dir()]; print('\\n'.join(p)); sys.exit(not p)"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE torch_library_output
  ERROR_VARIABLE error
  OUTPUT_STRIP_TRAILING_WHITESPACE
  TIMEOUT 30)
if(NOT result EQUAL 0 OR torch_library_output STREQUAL "")
  message(FATAL_ERROR
    "Cannot resolve pinned Torch library directories without importing "
    "Torch:\n${torch_library_output}\n${error}")
endif()
string(REPLACE "\n" ";" torch_library_directories
  "${torch_library_output}")

get_filename_component(ltj_library_directory
  "${canonical_ltj_library}" DIRECTORY)
get_filename_component(python_library_directory
  "${canonical_python_library}" DIRECTORY)
set(cann_library_directory "${canonical_cann_root}/lib64")
set(driver_library_directories
  "${canonical_driver_root}/lib64"
  "${canonical_driver_root}/lib64/common"
  "${canonical_driver_root}/lib64/driver")

set(runtime_library_directories)
foreach(directory IN ITEMS
    "${prefix_library_directory}"
    "${ltj_library_directory}"
    "${python_library_directory}"
    ${torch_library_directories}
    "${cann_library_directory}"
    ${driver_library_directories})
  if(directory STREQUAL "")
    message(FATAL_ERROR
      "Installed Ascend device LD_LIBRARY_PATH contains an empty entry")
  endif()
  if(NOT IS_ABSOLUTE "${directory}" OR NOT IS_DIRECTORY "${directory}")
    message(FATAL_ERROR
      "Installed Ascend device library directory is unavailable: ${directory}")
  endif()
  file(REAL_PATH "${directory}" canonical_directory)
  if(canonical_directory MATCHES ":")
    message(FATAL_ERROR
      "Installed Ascend device library directory contains ':': "
      "${canonical_directory}")
  endif()
  list(FIND runtime_library_directories "${canonical_directory}"
    existing_directory_index)
  if(existing_directory_index EQUAL -1)
    list(APPEND runtime_library_directories "${canonical_directory}")
  endif()
endforeach()
list(JOIN runtime_library_directories ":" runtime_library_path)
if(runtime_library_path STREQUAL "" OR
   runtime_library_path MATCHES "(^:|::$|::|:$)")
  message(FATAL_ERROR
    "Installed Ascend device LD_LIBRARY_PATH is empty or malformed: "
    "${runtime_library_path}")
endif()

set(device_environment
  "FLAGDNN_BACKEND_PATH=${prefix_library_directory}"
  "FLAGDNN_COMPILER_EXECUTABLE=${CODEGEN_PYTHON}"
  "FLAGDNN_ASCEND_PYTHON_MODULE_ROOT=${canonical_python_module_root}"
  FLAGDNN_ASCEND_REQUIRE_COMPILER_MODULES=1
  "FLAGDNN_LIBTRITON_JIT_LIBRARY=${canonical_ltj_library}"
  "FLAGDNN_TRITON_JIT_STANDALONE_COMPILER=${LTJ_STANDALONE}"
  "FLAGDNN_TRITON_JIT_CONFIG=${LTJ_CONFIG}"
  "FLAGDNN_TRITON_JIT_INCLUDE_DIRECTORY=${LTJ_INCLUDE}"
  "ASCEND_HOME_PATH=${canonical_cann_root}"
  "ASCEND_TOOLKIT_HOME=${canonical_cann_root}"
  TRITON_JIT_BACKEND=NPU
  TRITON_BACKEND=torch_npu
  TRITON_ALL_BLOCKS_PARALLEL=false
  LTJ_DUMP_KEY=1
  TORCH_DEVICE_BACKEND_AUTOLOAD=0
  PYTHONDONTWRITEBYTECODE=1
  PYTHONNOUSERSITE=1
  PYTHONHASHSEED=0
  PYTHONSAFEPATH=1
  "PYTHONPATH=${canonical_python_module_root}"
  "LD_LIBRARY_PATH=${runtime_library_path}"
  LC_ALL=C)
set(unset_environment
  --unset=LD_PRELOAD
  --unset=LD_AUDIT
  --unset=PYTHONHOME
  --unset=PYTHONPYCACHEPREFIX
  --unset=PYTHONUSERBASE
  --unset=FLAGDNN_ASCEND_RESOURCE_CONTAINMENT
  --unset=FLAGDNN_ASCEND_RESOURCE_ROOT
  --unset=FLAGDNN_CACHE_DIRECTORY
  --unset=TRITON_CACHE_DIR
  --unset=TMPDIR
  --unset=TRITON_ASCEND_ARCH
  --unset=TRITON_COMPILE_ONLY
  --unset=TRITON_ALWAYS_COMPILE
  --unset=TRITON_KERNEL_OVERRIDE
  --unset=TRITON_STORE_BINARY_ONLY
  --unset=TRITON_CACHE_MANAGER
  --unset=TRITON_REMOTE_CACHE_BACKEND
  --unset=TRITON_OVERRIDE_DIR
  --unset=FLAGDNN_ASCEND_CGROUP_PARENT
  --unset=FLAGDNN_ASCEND_PROVIDER_CGROUP
  --unset=FLAGDNN_ASCEND_SANDBOX_SUPERVISOR
  --unset=FLAGDNN_ASCEND_PROJECT_ID_RANGE
  --unset=FLAGDNN_ASCEND_QUOTA_HELPER)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    ${unset_environment}
    ${device_environment}
    -- "${consumer_executable}"
  RESULT_VARIABLE device_result
  OUTPUT_VARIABLE device_output
  ERROR_VARIABLE device_error
  TIMEOUT 600)

set(device_failure "")
if(NOT device_result EQUAL 0)
  string(APPEND device_failure
    "Installed Ascend trusted-local consumer failed with ${device_result}:\n"
    "${device_output}\n${device_error}\n")
endif()
string(REPLACE "\r\n" "\n" normalized_device_output "${device_output}")
string(STRIP "${normalized_device_output}" normalized_device_output)
set(expected_device_output
  "PASS installed FlagDNN Graph Add/Sub/Mul/Div/Min/Max/Identity/Neg/Elu/LeakyReLU/Abs/Ceil/Floor/ReLU/Reciprocal/Sqrt/Rsqrt/Exp/Log/Tanh/Sigmoid/Softplus/Swish/Gelu/GeluApproxTanh/Sin/Cos/Tan/Erf/Mod/Pow/SigmoidBackward/LogicalNot/LogicalAnd/LogicalOr/CmpEq/CmpNeq/CmpGt/CmpGe/CmpLt/CmpLe/BinarySelect/Reshape/Transpose/Slice/ReductionSum/ReductionAvg/ReductionMul/BatchNormInference/RMSNorm/LayerNorm/BatchNormTraining/MatMul/ConvFprop -> NPU libtriton_jit -> Ascend")
if(NOT normalized_device_output STREQUAL expected_device_output)
  string(APPEND device_failure
    "Installed Ascend trusted-local consumer stdout differs; expected "
    "'${expected_device_output}', got '${normalized_device_output}'.\n")
endif()
string(REGEX MATCHALL "\\[LTJ_CACHE_MISS\\]" ltj_cache_misses
  "${device_error}")
list(LENGTH ltj_cache_misses ltj_cache_miss_count)
# This is a fresh executable and therefore starts with an empty LTJ in-memory
# overload map.  The sixty-two-stage autotuned graph has two immutable
# candidates per stage.  Five stages reuse existing in-process overloads, so
# the remaining fifty-seven unique stages produce exactly one hundred fourteen
# misses during build.  No misses during execute proves both candidate
# prewarming and the selected-candidate execute path use the installed LTJ
# cache contract.
set(expected_ltj_cache_miss_count 114)
if(NOT ltj_cache_miss_count EQUAL expected_ltj_cache_miss_count)
  string(APPEND device_failure
    "Installed Ascend trusted-local consumer expected exactly "
    "${expected_ltj_cache_miss_count} fresh-process LTJ cache misses, observed "
    "${ltj_cache_miss_count}:\n${device_error}\n")
endif()
foreach(marker IN ITEMS
    BEGIN_CREATE END_CREATE BEGIN_EXECUTE END_EXECUTE)
  string(REGEX MATCHALL "${marker}" marker_matches "${device_error}")
  list(LENGTH marker_matches marker_count)
  if(NOT marker_count EQUAL 1)
    string(APPEND device_failure
      "Installed Ascend trusted-local consumer expected exactly one ${marker} "
      "marker, observed ${marker_count}:\n${device_error}\n")
  endif()
  string(FIND "${device_error}" "${marker}" "${marker}_offset")
endforeach()
string(FIND "${device_error}" "[LTJ_CACHE_MISS]" ltj_cache_miss_offset)
string(FIND "${device_error}" "[LTJ_CACHE_MISS]" ltj_last_cache_miss_offset
  REVERSE)
if(NOT BEGIN_CREATE_offset LESS END_CREATE_offset OR
   NOT END_CREATE_offset LESS BEGIN_EXECUTE_offset OR
   NOT BEGIN_EXECUTE_offset LESS END_EXECUTE_offset OR
   NOT BEGIN_CREATE_offset LESS ltj_cache_miss_offset OR
   NOT ltj_cache_miss_offset LESS END_CREATE_offset OR
   NOT BEGIN_CREATE_offset LESS ltj_last_cache_miss_offset OR
   NOT ltj_last_cache_miss_offset LESS END_CREATE_offset)
  string(APPEND device_failure
    "Installed Ascend trusted-local consumer markers do not prove that the "
    "${expected_ltj_cache_miss_count} LTJ misses occurred during "
    "create/prewarm and not execute:\n"
    "${device_error}\n")
endif()
if(NOT device_failure STREQUAL "")
  message(FATAL_ERROR "${device_failure}")
endif()

message(STATUS
  "Installed Ascend trusted-local consumer PASS with caller-owned ACL lifecycle")
