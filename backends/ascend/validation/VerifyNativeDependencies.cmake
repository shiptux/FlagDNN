# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

if(NOT EXISTS "${CORE_LIBRARY}" OR NOT EXISTS "${BACKEND_LIBRARY}")
  message(FATAL_ERROR "Ascend dependency-boundary inputs do not exist")
endif()
find_program(READELF_EXECUTABLE readelf REQUIRED)

execute_process(COMMAND "${READELF_EXECUTABLE}" -d "${CORE_LIBRARY}"
  RESULT_VARIABLE core_status OUTPUT_VARIABLE core_dynamic
  ERROR_VARIABLE core_error)
if(NOT core_status EQUAL 0)
  message(FATAL_ERROR "Cannot inspect FlagDNN core: ${core_error}")
endif()
if(core_dynamic MATCHES "libascendcl|libruntime|libnnopbase|libopapi")
  message(FATAL_ERROR "FlagDNN core directly depends on a CANN library")
endif()

execute_process(COMMAND "${READELF_EXECUTABLE}" -d "${BACKEND_LIBRARY}"
  RESULT_VARIABLE backend_status OUTPUT_VARIABLE backend_dynamic
  ERROR_VARIABLE backend_error)
if(NOT backend_status EQUAL 0)
  message(FATAL_ERROR "Cannot inspect Ascend backend: ${backend_error}")
endif()
if(NOT backend_dynamic MATCHES "libascendcl")
  message(FATAL_ERROR "Ascend backend has no direct AscendCL dependency")
endif()
if(backend_dynamic MATCHES "libnnopbase|libopapi|torch_npu")
  message(FATAL_ERROR
    "Production Ascend backend directly depends on a validation library")
endif()
