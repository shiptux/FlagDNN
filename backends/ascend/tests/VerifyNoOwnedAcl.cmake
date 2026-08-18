# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED BACKEND_LIBRARY OR NOT EXISTS "${BACKEND_LIBRARY}")
  message(FATAL_ERROR "BACKEND_LIBRARY does not name the Ascend plugin")
endif()
if(NOT DEFINED LTJ_LIBRARY OR NOT EXISTS "${LTJ_LIBRARY}")
  message(FATAL_ERROR "LTJ_LIBRARY does not name the pinned TritonJIT DSO")
endif()
if(NOT CMAKE_NM)
  find_program(CMAKE_NM nm REQUIRED)
endif()
execute_process(
  COMMAND "${CMAKE_NM}" -D --undefined-only "${BACKEND_LIBRARY}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE symbols
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Cannot inspect Ascend plugin symbols: ${error}")
endif()

foreach(forbidden IN ITEMS
    aclInit aclFinalize aclFinalizeReference aclrtSetDevice
    aclrtCreateContext aclrtDestroyContext aclrtResetDevice
    aclrtResetDeviceForce)
  if(symbols MATCHES "(^|[ \t\n])${forbidden}(@|[ \t\n]|$)")
    message(FATAL_ERROR
      "Ascend plugin must not reference caller-owned lifecycle API ${forbidden}")
  endif()
endforeach()

# The public raw-launch method is inline in the installed header even though
# libtriton_jit explicitly instantiates it.  The plugin must call the exported
# instantiation rather than copy NpuBackend lifecycle code into its own DSO.
set(raw_launch_symbol
  "_ZNK10triton_jit21TritonJITFunctionImplINS_10NpuBackendEE20launch_with_raw_argsEPvjjjjjNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPS3_m")
if(NOT symbols MATCHES
   "(^|[ \t\n])U[ \t]+${raw_launch_symbol}(@|[ \t\n]|$)")
  message(FATAL_ERROR
    "Ascend plugin must import the pinned out-of-line LTJ raw-launch symbol")
endif()

execute_process(
  COMMAND "${CMAKE_NM}" -D --defined-only "${BACKEND_LIBRARY}"
  RESULT_VARIABLE backend_defined_result
  OUTPUT_VARIABLE backend_defined_symbols
  ERROR_VARIABLE backend_defined_error)
if(NOT backend_defined_result EQUAL 0)
  message(FATAL_ERROR
    "Cannot inspect defined Ascend plugin symbols: ${backend_defined_error}")
endif()
if(backend_defined_symbols MATCHES "${raw_launch_symbol}")
  message(FATAL_ERROR
    "Ascend plugin must not define or inline-export LTJ raw launch")
endif()

execute_process(
  COMMAND "${CMAKE_NM}" -D --defined-only "${LTJ_LIBRARY}"
  RESULT_VARIABLE ltj_defined_result
  OUTPUT_VARIABLE ltj_defined_symbols
  ERROR_VARIABLE ltj_defined_error)
if(NOT ltj_defined_result EQUAL 0)
  message(FATAL_ERROR
    "Cannot inspect pinned TritonJIT symbols: ${ltj_defined_error}")
endif()
if(NOT ltj_defined_symbols MATCHES
   "(^|[ \t\n])[TW][ \t]+${raw_launch_symbol}(@|[ \t\n]|$)")
  message(FATAL_ERROR
    "Pinned TritonJIT does not export the required raw-launch instantiation")
endif()

if(NOT CMAKE_READELF)
  find_program(CMAKE_READELF readelf REQUIRED)
endif()
execute_process(
  COMMAND "${CMAKE_READELF}" -d "${BACKEND_LIBRARY}"
  RESULT_VARIABLE readelf_result
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE readelf_error)
if(NOT readelf_result EQUAL 0)
  message(FATAL_ERROR
    "Cannot inspect Ascend plugin dependencies: ${readelf_error}")
endif()

foreach(required_pattern IN ITEMS
    "Shared library: \\[libascendcl\\.so"
    "Shared library: \\[libtriton_jit\\.so"
    "Shared library: \\[libpython[0-9]+\\.[0-9]+\\.so")
  if(NOT dynamic_section MATCHES "${required_pattern}")
    message(FATAL_ERROR
      "Ascend plugin is missing required direct dependency matching "
      "${required_pattern}")
  endif()
endforeach()

foreach(forbidden_pattern IN ITEMS
    "Shared library: \\[libnnopbase"
    "Shared library: \\[libopapi"
    "Shared library: \\[libtorch_npu")
  if(dynamic_section MATCHES "${forbidden_pattern}")
    message(FATAL_ERROR
      "Ascend plugin must not directly depend on ${forbidden_pattern}")
  endif()
endforeach()
