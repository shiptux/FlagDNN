# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED SOURCE_FILE OR NOT EXISTS "${SOURCE_FILE}")
  message(FATAL_ERROR
    "JIT candidate compatibility contract requires an existing SOURCE_FILE")
endif()
if(NOT DEFINED HCU_ERROR_HEADER OR NOT EXISTS "${HCU_ERROR_HEADER}")
  message(FATAL_ERROR
    "JIT candidate compatibility contract requires HCU_ERROR_HEADER")
endif()
if(NOT DEFINED HCU_BACKEND_HEADER OR NOT EXISTS "${HCU_BACKEND_HEADER}")
  message(FATAL_ERROR
    "JIT candidate compatibility contract requires HCU_BACKEND_HEADER")
endif()
if(NOT DEFINED HCU_JIT_UTILS_HEADER OR
   NOT EXISTS "${HCU_JIT_UTILS_HEADER}")
  message(FATAL_ERROR
    "JIT candidate compatibility contract requires HCU_JIT_UTILS_HEADER")
endif()

file(READ "${SOURCE_FILE}" _flagdnn_hygon_jit_source)
string(FIND "${_flagdnn_hygon_jit_source}"
  "bool is_candidate_compatibility_error("
  _flagdnn_classifier_begin)
string(FIND "${_flagdnn_hygon_jit_source}"
  "\nclass ScopedEnvironmentVariable" _flagdnn_classifier_end)
if(_flagdnn_classifier_begin EQUAL -1 OR
   _flagdnn_classifier_end EQUAL -1 OR
   _flagdnn_classifier_end LESS_EQUAL _flagdnn_classifier_begin)
  message(FATAL_ERROR
    "Cannot isolate the Hygon JIT candidate compatibility classifier")
endif()

math(EXPR _flagdnn_classifier_length
  "${_flagdnn_classifier_end} - ${_flagdnn_classifier_begin}")
string(SUBSTRING "${_flagdnn_hygon_jit_source}"
  ${_flagdnn_classifier_begin} ${_flagdnn_classifier_length}
  _flagdnn_classifier)
string(REGEX REPLACE "[ \t\r\n]+" "" _flagdnn_classifier
  "${_flagdnn_classifier}")

# Candidate rejection is the only autotune path allowed to continue after an
# exception. Require typed Hygon/libtriton_jit errors, an exact HIP whitelist,
# and the HCU shared-memory semantic reason. Architecture, metadata and generic
# std::exception failures remain hard failures; message matching is forbidden.
foreach(_required IN ITEMS
    "dynamic_cast<constHygonError*>(&error)"
    "dynamic_cast<consttriton_jit::HcuError*>(&error)"
    "triton_jit::HcuErrorKind::kSharedMemoryLimit"
    "triton_jit::HcuErrorKind::kRuntimeApi"
    "hcu_error->has_hip_result()"
    "hipErrorLaunchOutOfResources"
    "hipErrorInvalidConfiguration"
    "hipErrorInvalidDeviceFunction"
    "hipErrorInvalidImage"
    "hipErrorNoBinaryForGpu"
    "hipErrorInvalidKernelFile")
  string(FIND "${_flagdnn_classifier}" "${_required}" _required_index)
  if(_required_index EQUAL -1)
    message(FATAL_ERROR
      "Hygon candidate classifier is missing typed rule: ${_required}")
  endif()
endforeach()
foreach(_forbidden IN ITEMS
    "what()"
    "std::string_view"
    "std::string("
    "find("
    "strstr("
    "kArchitectureMismatch")
  string(FIND "${_flagdnn_classifier}" "${_forbidden}" _forbidden_index)
  if(NOT _forbidden_index EQUAL -1)
    message(FATAL_ERROR
      "Hygon candidate classifier contains forbidden message/architecture "
      "recovery: ${_forbidden}")
  endif()
endforeach()

file(READ "${HCU_ERROR_HEADER}" _hcu_error_header)
foreach(_required IN ITEMS
    "class TRITON_JIT_HCU_API HcuError final"
    "~HcuError() override;"
    "kRuntimeApi"
    "kArchitectureMismatch"
    "kSharedMemoryLimit"
    "throw_hcu_runtime_error"
    "throw_hcu_error")
  string(FIND "${_hcu_error_header}" "${_required}" _required_index)
  if(_required_index EQUAL -1)
    message(FATAL_ERROR
      "libtriton_jit HCU typed-error ABI is missing: ${_required}")
  endif()
endforeach()

file(READ "${HCU_BACKEND_HEADER}" _hcu_backend_header)
foreach(_required IN ITEMS
    "throw_hcu_runtime_error("
    "HcuErrorKind::kArchitectureMismatch"
    "HcuErrorKind::kSharedMemoryLimit")
  string(FIND "${_hcu_backend_header}" "${_required}" _required_index)
  if(_required_index EQUAL -1)
    message(FATAL_ERROR
      "libtriton_jit HCU backend does not preserve typed errors: ${_required}")
  endif()
endforeach()

file(READ "${HCU_JIT_UTILS_HEADER}" _hcu_jit_utils_header)
string(FIND "${_hcu_jit_utils_header}"
  "detail::throw_hcu_runtime_error(code, error_detail);"
  _hcu_check_index)
if(_hcu_check_index EQUAL -1)
  message(FATAL_ERROR
    "libtriton_jit checkHcuErrors loses the HCU runtime result")
endif()

message(STATUS "Hygon JIT candidate compatibility contract passed")
