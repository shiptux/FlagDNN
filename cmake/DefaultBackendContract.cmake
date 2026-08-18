# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED SOURCE_ROOT)
  message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${SOURCE_ROOT}/cmake/FlagDNNDefaultBackend.cmake")

if(DEFINED FLAGDNN_DEFAULT_BACKEND_CONTRACT_INVALID)
  flagdnn_resolve_default_backend(_unused "hygon" "Vendor-X")
  message(FATAL_ERROR "the invalid explicit default unexpectedly succeeded")
endif()

flagdnn_resolve_default_backend(_actual "hygon;nvidia" auto)
if(NOT _actual STREQUAL "nvidia")
  message(FATAL_ERROR
    "auto must preserve NVIDIA as the multi-backend compatibility default")
endif()

flagdnn_resolve_default_backend(_actual "nvidia;hygon" auto)
if(NOT _actual STREQUAL "nvidia")
  message(FATAL_ERROR "auto default changed with backend registration order")
endif()

flagdnn_resolve_default_backend(_actual "hygon" auto)
if(NOT _actual STREQUAL "hygon")
  message(FATAL_ERROR "auto did not select the sole configured backend")
endif()

flagdnn_resolve_default_backend(_actual "hygon;nvidia" hygon)
if(NOT _actual STREQUAL "hygon")
  message(FATAL_ERROR "a valid explicit default was not honored")
endif()

flagdnn_resolve_default_backend(_actual "" auto)
if(NOT _actual STREQUAL "nvidia")
  message(FATAL_ERROR "core-only fallback is not deterministic")
endif()

flagdnn_resolve_default_backend(_actual "" vendorx)
if(NOT _actual STREQUAL "vendorx")
  message(FATAL_ERROR
    "core-only build rejected a default plugin supplied at runtime")
endif()

flagdnn_resolve_default_backend(_actual "hygon" vendorx)
if(NOT _actual STREQUAL "vendorx")
  message(FATAL_ERROR
    "built-in plugins prevented selecting an external default plugin")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DSOURCE_ROOT=${SOURCE_ROOT}"
    -DFLAGDNN_DEFAULT_BACKEND_CONTRACT_INVALID=ON
    -P "${CMAKE_CURRENT_LIST_FILE}"
  RESULT_VARIABLE _invalid_result
  OUTPUT_VARIABLE _invalid_stdout
  ERROR_VARIABLE _invalid_stderr)
if(_invalid_result EQUAL 0)
  message(FATAL_ERROR "an explicit default absent from FLAGDNN_BACKENDS passed")
endif()
set(_invalid_output "${_invalid_stdout}\n${_invalid_stderr}")
if(NOT _invalid_output MATCHES
   "FLAGDNN_DEFAULT_BACKEND must be auto or match")
  message(FATAL_ERROR
    "invalid-default diagnostic changed unexpectedly: ${_invalid_output}")
endif()
