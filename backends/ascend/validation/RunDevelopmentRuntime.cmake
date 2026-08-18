# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)

foreach(required IN ITEMS
    EXECUTABLE COMPILER_EXECUTABLE COMPILER_ENTRY BACKEND_PATH)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR
      "RunDevelopmentRuntime requires ${required}")
  endif()
endforeach()

if(NOT DEFINED MODE OR "${MODE}" STREQUAL "")
  set(MODE cache)
endif()
if(NOT MODE STREQUAL "cache" AND
   NOT MODE STREQUAL "resource_budget" AND
   NOT MODE STREQUAL "resource_graph_lifecycle" AND
   NOT MODE STREQUAL "resource_dual_handle" AND
   NOT MODE STREQUAL "resource_cache_snapshot_host")
  message(FATAL_ERROR
    "RunDevelopmentRuntime does not recognize mode ${MODE}")
endif()
if(MODE STREQUAL "resource_cache_snapshot_host")
  set(runtime_timeout 60)
elseif(MODE STREQUAL "cache")
  set(runtime_timeout 600)
else()
  set(runtime_timeout 1200)
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "FLAGDNN_BACKEND_PATH=${BACKEND_PATH}"
    LTJ_DUMP_KEY=1
    -- "${EXECUTABLE}" "${MODE}" "${COMPILER_EXECUTABLE}" "${COMPILER_ENTRY}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
  TIMEOUT "${runtime_timeout}")
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "Ascend development ${MODE} integration failed with ${result}:\n"
    "${output}\n${error}")
endif()

if(MODE STREQUAL "resource_cache_snapshot_host")
  string(REGEX MATCHALL
    "ASCEND_DEVELOPMENT_CACHE_SNAPSHOT_HOST PASS files=2 bytes=17 mutation_detected=1 symlink_rejected=1"
    host_snapshot_passes "${output}")
  list(LENGTH host_snapshot_passes host_snapshot_pass_count)
  if(NOT host_snapshot_pass_count EQUAL 1)
    message(FATAL_ERROR
      "Ascend host cache-snapshot self-test omitted its exact PASS contract:\n"
      "${output}\n${error}")
  endif()
  message(STATUS "Ascend host cache-snapshot no-follow identity PASS")
  return()
endif()

if(MODE STREQUAL "cache")
  set(expected_contract "ASCEND_DEVELOPMENT_CACHE PASS executes=16")
elseif(MODE STREQUAL "resource_budget")
  set(expected_contract
    "ASCEND_DEVELOPMENT_RESOURCE_BUDGET PASS warmup=1000 windows=6 iterations_per_window=10000")
elseif(MODE STREQUAL "resource_graph_lifecycle")
  set(expected_contract
    "ASCEND_DEVELOPMENT_RESOURCE_GRAPH_LIFECYCLE_BUDGET PASS warmup_lifecycles=20 measured_lifecycles=100 executes_per_lifecycle=4 sample_interval=10")
else()
  set(expected_contract
    "ASCEND_DEVELOPMENT_RESOURCE_DUAL_HANDLE_BUDGET PASS handles=2 warmup_per_handle=1000 windows=6 executes_per_handle_per_window=5000")
endif()

if(NOT output MATCHES "${expected_contract}" OR
   NOT output MATCHES
   "ASCEND_DEVELOPMENT_RUNTIME PASS mode=${MODE} caller_acl_reference_count=0")
  message(FATAL_ERROR
    "Ascend development ${MODE} integration omitted its PASS contract:\n"
    "${output}\n${error}")
endif()

if(NOT MODE STREQUAL "cache")
  set(all_snapshot_pattern
    "ASCEND_DEVELOPMENT_CACHE_SNAPSHOT mode=${MODE} phase=(before|after) scope=(graph|production) files=[0-9]+ bytes=[0-9]+ manifest_sha256=[0-9a-f]+ root_device=[0-9]+ root_inode=[0-9]+")
  string(REGEX MATCHALL "${all_snapshot_pattern}" all_cache_snapshots
    "${output}")
  list(LENGTH all_cache_snapshots all_cache_snapshot_count)
  if(NOT all_cache_snapshot_count EQUAL 4)
    message(FATAL_ERROR
      "Ascend development ${MODE} expected exactly four cache snapshots, "
      "observed ${all_cache_snapshot_count}:\n${output}\n${error}")
  endif()

  foreach(cache_scope IN ITEMS graph production)
    foreach(snapshot_phase IN ITEMS before after)
      set(snapshot_pattern
        "ASCEND_DEVELOPMENT_CACHE_SNAPSHOT mode=${MODE} phase=${snapshot_phase} scope=${cache_scope} files=([0-9]+) bytes=([0-9]+) manifest_sha256=([0-9a-f]+) root_device=([0-9]+) root_inode=([0-9]+)")
      string(REGEX MATCHALL "${snapshot_pattern}" snapshot_matches
        "${output}")
      list(LENGTH snapshot_matches snapshot_match_count)
      if(NOT snapshot_match_count EQUAL 1)
        message(FATAL_ERROR
          "Ascend development ${MODE} expected one ${snapshot_phase} "
          "${cache_scope} cache snapshot, observed ${snapshot_match_count}:\n"
          "${output}\n${error}")
      endif()
      string(REGEX MATCH "${snapshot_pattern}" snapshot_match "${output}")
      set(snapshot_prefix "snapshot_${snapshot_phase}_${cache_scope}")
      set("${snapshot_prefix}_files" "${CMAKE_MATCH_1}")
      set("${snapshot_prefix}_bytes" "${CMAKE_MATCH_2}")
      set("${snapshot_prefix}_digest" "${CMAKE_MATCH_3}")
      set("${snapshot_prefix}_device" "${CMAKE_MATCH_4}")
      set("${snapshot_prefix}_inode" "${CMAKE_MATCH_5}")
      string(LENGTH "${CMAKE_MATCH_3}" snapshot_digest_length)
      if(NOT snapshot_digest_length EQUAL 64)
        message(FATAL_ERROR
          "Ascend development ${MODE} ${snapshot_phase} ${cache_scope} "
          "cache snapshot has a non-SHA-256 manifest digest:\n${output}")
      endif()
    endforeach()

    foreach(snapshot_field IN ITEMS files bytes digest device inode)
      set(before_variable "snapshot_before_${cache_scope}_${snapshot_field}")
      set(after_variable "snapshot_after_${cache_scope}_${snapshot_field}")
      if(NOT "${${before_variable}}" STREQUAL "${${after_variable}}")
        message(FATAL_ERROR
          "Ascend development ${MODE} ${cache_scope} cache ${snapshot_field} "
          "changed across measured execute windows:\n${output}\n${error}")
      endif()
    endforeach()

    set(before_files_variable "snapshot_before_${cache_scope}_files")
    set(before_bytes_variable "snapshot_before_${cache_scope}_bytes")
    set(before_digest_variable "snapshot_before_${cache_scope}_digest")
    set(unchanged_contract
      "ASCEND_DEVELOPMENT_CACHE_UNCHANGED PASS mode=${MODE} scope=${cache_scope} files=${${before_files_variable}} bytes=${${before_bytes_variable}} manifest_sha256=${${before_digest_variable}}")
    string(REGEX MATCHALL "${unchanged_contract}" unchanged_matches
      "${output}")
    list(LENGTH unchanged_matches unchanged_match_count)
    if(NOT unchanged_match_count EQUAL 1)
      message(FATAL_ERROR
        "Ascend development ${MODE} omitted the exact ${cache_scope} "
        "cache-unchanged contract:\n${output}\n${error}")
    endif()
  endforeach()
endif()

string(REGEX MATCHALL "\\[LTJ_CACHE_MISS\\]" cache_misses "${error}")
list(LENGTH cache_misses cache_miss_count)
if(NOT cache_miss_count EQUAL 1)
  message(FATAL_ERROR
    "Ascend development ${MODE} integration expected one create-time LTJ "
    "miss, observed ${cache_miss_count}:\n${error}")
endif()

foreach(marker IN ITEMS BEGIN_CREATE END_CREATE BEGIN_EXECUTE END_EXECUTE)
  string(REGEX MATCHALL "${marker}" marker_matches "${error}")
  list(LENGTH marker_matches marker_count)
  if(NOT marker_count EQUAL 1)
    message(FATAL_ERROR
      "Ascend development ${MODE} integration expected one ${marker}, "
      "observed ${marker_count}:\n${error}")
  endif()
  string(FIND "${error}" "${marker}" "${marker}_offset")
endforeach()
string(FIND "${error}" "[LTJ_CACHE_MISS]" cache_miss_offset)
if(NOT BEGIN_CREATE_offset LESS cache_miss_offset OR
   NOT cache_miss_offset LESS END_CREATE_offset OR
   NOT END_CREATE_offset LESS BEGIN_EXECUTE_offset OR
   NOT BEGIN_EXECUTE_offset LESS END_EXECUTE_offset)
  message(FATAL_ERROR
    "Ascend development ${MODE} markers do not prove create-time prewarm and "
    "execute-time overload reuse:\n${error}")
endif()

string(TOLOWER "${error}" lower_error)
foreach(failure_text IN ITEMS
    "workspace cleanup callback"
    "rtmalloc workspace failed"
    "allocation failed")
  string(FIND "${lower_error}" "${failure_text}" failure_offset)
  if(NOT failure_offset EQUAL -1)
    message(FATAL_ERROR
      "Ascend development ${MODE} reported ${failure_text}:\n${error}")
  endif()
endforeach()

message(STATUS
  "Ascend development ${MODE} measured output:\n${output}")
message(STATUS
  "Ascend development ${MODE} create/prewarm miss and execute cache-hit PASS")
