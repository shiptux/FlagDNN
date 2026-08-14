# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED SOURCE_FILE OR NOT EXISTS "${SOURCE_FILE}")
  message(FATAL_ERROR
    "Hygon JIT global-state contract requires an existing SOURCE_FILE")
endif()

file(READ "${SOURCE_FILE}" _flagdnn_hygon_jit_source)

function(_flagdnn_require_text haystack needle description)
  string(FIND "${haystack}" "${needle}" _flagdnn_match)
  if(_flagdnn_match EQUAL -1)
    message(FATAL_ERROR "${description}")
  endif()
endfunction()

function(_flagdnn_isolate begin_marker end_marker output)
  string(FIND "${_flagdnn_hygon_jit_source}" "${begin_marker}" _begin)
  string(FIND "${_flagdnn_hygon_jit_source}" "${end_marker}" _end)
  if(_begin EQUAL -1 OR _end EQUAL -1 OR _end LESS_EQUAL _begin)
    message(FATAL_ERROR
      "Cannot isolate Hygon JIT global-state section: ${begin_marker}")
  endif()
  math(EXPR _length "${_end} - ${_begin}")
  string(SUBSTRING "${_flagdnn_hygon_jit_source}"
    ${_begin} ${_length} _section)
  set(${output} "${_section}" PARENT_SCOPE)
endfunction()

# A foreign backend is an embedding-application conflict, not a value the
# FlagDNN Hygon plugin may silently replace. Both the C environment and an
# already initialized interpreter must be inspected by a read-only entry point.
_flagdnn_isolate(
  "void verify_triton_jit_backend_compatibility() {"
  "\nbool is_candidate_compatibility_error"
  _compatibility)
_flagdnn_require_text("${_compatibility}"
  "std::getenv(\"TRITON_JIT_BACKEND\")"
  "Hygon JIT does not inspect the C TRITON_JIT_BACKEND")
_flagdnn_require_text("${_compatibility}"
  "active_python_triton_jit_backend()"
  "Hygon JIT does not inspect Python os.environ read-only")
_flagdnn_require_text("${_compatibility}"
  "FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED"
  "Hygon JIT backend conflicts are not compilation failures")
_flagdnn_require_text("${_compatibility}"
  "kHygonTritonJitBackend"
  "Hygon JIT backend compatibility is not exact")

# Python initialization itself does not publish a backend selection. The
# selection belongs to the engine-build scope below.
_flagdnn_isolate(
  "void initialize_python_runtime() {"
  "\nstruct TuningAllocation"
  _runtime_initialization)
string(FIND "${_runtime_initialization}" "setenv(" _runtime_set)
string(FIND "${_runtime_initialization}"
  "active_python_triton_jit_backend(true)" _runtime_python_set)
if(NOT _runtime_set EQUAL -1 OR NOT _runtime_python_set EQUAL -1)
  message(FATAL_ERROR "Python initialization permanently selects a JIT backend")
endif()

# Environment changes made for libtriton_jit must update both the native and
# Python views, remember both original values, and restore them on every exit.
_flagdnn_isolate(
  "class ScopedEnvironmentVariable {"
  "\nstd::string read_regular_file"
  _scoped_environment)
foreach(required_text IN ITEMS
    "original_c_"
    "original_python_"
    "PyMapping_SetItemString(environment, name_.c_str(), replacement)"
    "PyMapping_DelItemString(environment, name_.c_str())"
    "setenv(name_.c_str(), original_c_->c_str(), 1)"
    "unsetenv(name_.c_str())")
  _flagdnn_require_text("${_scoped_environment}" "${required_text}"
    "Hygon JIT scoped environment does not preserve/restore both views")
endforeach()

# The configured Python paths and libtriton_jit script directory share this
# helper. It must not append a duplicate on a later engine build.
_flagdnn_isolate(
  "void configure_active_python_path("
  "\nvoid configure_python_path()"
  _active_python_path)
_flagdnn_require_text("${_active_python_path}"
  "PySequence_Contains(sys_path, value)"
  "Hygon JIT sys.path insertion has no duplicate guard")
_flagdnn_require_text("${_active_python_path}"
  "present == 0 && PyList_Insert(sys_path, 0, value)"
  "Hygon JIT sys.path insertion does not honor its duplicate guard")
_flagdnn_isolate(
  "class ScopedPythonPath {"
  "\nvoid configure_python_path()"
  _scoped_python_path)
_flagdnn_require_text("${_scoped_python_path}"
  "snapshot_ = PySequence_List(original_)"
  "Hygon JIT does not snapshot the embedding application's sys.path")
_flagdnn_require_text("${_scoped_python_path}"
  "PyList_SetSlice(original_, 0, PyList_Size(original_), snapshot_)"
  "Hygon JIT does not restore the embedding application's sys.path contents")
_flagdnn_require_text("${_scoped_python_path}"
  "PyObject_SetAttrString(sys, \"path\", original_)"
  "Hygon JIT does not restore the embedding application's sys.path")
_flagdnn_isolate(
  "void configure_python_path() {"
  "\nvoid initialize_python_runtime()"
  _configured_python_path)
string(FIND "${_configured_python_path}" "setenv(" _pythonpath_setenv)
string(FIND "${_configured_python_path}" "std::getenv(\"PYTHONPATH\")"
  _pythonpath_read)
if(NOT _pythonpath_setenv EQUAL -1 OR NOT _pythonpath_read EQUAL -1)
  message(FATAL_ERROR "Hygon JIT permanently mutates or consumes PYTHONPATH")
endif()

# The first read-only compatibility check precedes the scoped backend
# selection. Every mutation object must outlive compilation and restore state.
_flagdnn_isolate(
  "std::unique_lock lock(libtriton_jit_mutex);"
  "require(!artifact.stages.empty()"
  _engine_setup)
string(FIND "${_engine_setup}"
  "verify_triton_jit_backend_compatibility();" _setup_verify)
string(FIND "${_engine_setup}"
  "promote_python_runtime();" _setup_promote)
string(FIND "${_engine_setup}"
  "initialize_python_runtime();" _setup_runtime)
string(FIND "${_engine_setup}"
  "const ScopedEnvironmentVariable backend_selection(" _setup_backend_scope)
string(FIND "${_engine_setup}"
  "const ScopedPythonPath python_path;" _setup_path_scope)
string(FIND "${_engine_setup}"
  "configure_python_path();" _setup_python_path)
string(FIND "${_engine_setup}"
  "configure_active_python_path(" _setup_sys_path)
if(_setup_verify EQUAL -1 OR _setup_promote EQUAL -1 OR
   _setup_runtime EQUAL -1 OR _setup_backend_scope EQUAL -1 OR
   _setup_path_scope EQUAL -1 OR _setup_python_path EQUAL -1 OR
   _setup_sys_path EQUAL -1 OR
   _setup_verify GREATER _setup_promote OR
   _setup_verify GREATER _setup_runtime OR
   _setup_runtime GREATER _setup_backend_scope OR
   _setup_backend_scope GREATER _setup_path_scope OR
   _setup_path_scope GREATER _setup_python_path OR
   _setup_python_path GREATER _setup_sys_path)
  message(FATAL_ERROR
    "Hygon JIT global-state verification/initialization order regressed")
endif()

# A prepared standalone helper is part of the compiler, not merely an import
# detail. Its verified build identity must partition Triton's persistent cache
# so a compatibility rewrite cannot reuse code produced by an older helper.
_flagdnn_isolate(
  "device_triton_cache_directory("
  "\nvoid promote_python_runtime()"
  _device_cache)
_flagdnn_require_text("${_device_cache}"
  "flagdnn-hygon-v2-"
  "Hygon Triton cache namespace has no explicit schema version")
_flagdnn_require_text("${_device_cache}"
  "FLAGDNN_LIBTRITON_JIT_BUILD_IDENTITY"
  "Hygon Triton cache namespace omits the verified JIT/helper identity")
_flagdnn_require_text("${_device_cache}"
  "context.device_identity"
  "Hygon Triton cache namespace omits the device/compiler identity")

message(STATUS "Hygon JIT global-state contract passed")
