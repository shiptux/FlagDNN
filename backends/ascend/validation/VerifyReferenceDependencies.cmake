# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

if(NOT EXISTS "${REFERENCE_EXECUTABLE}")
  message(FATAL_ERROR "Ascend reference executable does not exist")
endif()
find_program(READELF_EXECUTABLE readelf REQUIRED)
execute_process(COMMAND "${READELF_EXECUTABLE}" -d "${REFERENCE_EXECUTABLE}"
  RESULT_VARIABLE reference_status OUTPUT_VARIABLE reference_dynamic
  ERROR_VARIABLE reference_error)
if(NOT reference_status EQUAL 0)
  message(FATAL_ERROR
    "Cannot inspect Ascend reference executable: ${reference_error}")
endif()
if(NOT reference_dynamic MATCHES "libnnopbase")
  message(FATAL_ERROR
    "Ascend reference executable does not directly need libnnopbase")
endif()
if(NOT reference_dynamic MATCHES "libopapi_math")
  message(FATAL_ERROR
    "Ascend reference executable does not directly need libopapi_math")
endif()
if(REFERENCE_REQUIRES_OPAPI_NN AND
   NOT reference_dynamic MATCHES "libopapi_nn")
  message(FATAL_ERROR
    "Ascend neural-network reference does not directly need libopapi_nn")
elseif(NOT REFERENCE_REQUIRES_OPAPI_NN AND
       reference_dynamic MATCHES "libopapi_nn")
  message(FATAL_ERROR
    "Ascend math-only reference unexpectedly needs libopapi_nn")
endif()
if(reference_dynamic MATCHES "libopapi\\.so|torch_npu")
  message(FATAL_ERROR
    "Ascend reference executable uses a forbidden aggregate/framework library")
endif()
