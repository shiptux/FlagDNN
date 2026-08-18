include_guard(GLOBAL)

function(flagdnn_add_backend_plugin backend_name)
  set(options INSTALL)
  set(one_value_arguments ABI_VERSION)
  set(multi_value_arguments SOURCES INCLUDE_DIRECTORIES LINK_LIBRARIES)
  cmake_parse_arguments(
    FLAGDNN_BACKEND
    "${options}"
    "${one_value_arguments}"
    "${multi_value_arguments}"
    ${ARGN})

  string(LENGTH "${backend_name}" backend_name_length)
  if(NOT backend_name MATCHES "^[a-z][a-z0-9_]*$" OR
     backend_name_length GREATER 63)
    message(FATAL_ERROR
      "FlagDNN backend name '${backend_name}' is not loader-safe")
  endif()
  if(NOT FLAGDNN_BACKEND_SOURCES)
    message(FATAL_ERROR
      "flagdnn_add_backend_plugin(${backend_name}) requires SOURCES")
  endif()
  if(NOT FLAGDNN_BACKEND_ABI_VERSION)
    set(FLAGDNN_BACKEND_ABI_VERSION 2)
  endif()
  if(NOT FLAGDNN_BACKEND_ABI_VERSION MATCHES "^[23]$")
    message(FATAL_ERROR
      "FlagDNN backend ABI version must be 2 or 3")
  endif()
  if(FLAGDNN_BACKEND_ABI_VERSION EQUAL 3 AND
     NOT backend_name STREQUAL "ascend")
    message(FATAL_ERROR
      "FlagDNN backend ABI version 3 is currently reserved for ascend")
  endif()

  set(target "flagdnn_backend_${backend_name}")
  if(TARGET "${target}")
    message(FATAL_ERROR "FlagDNN backend target '${target}' already exists")
  endif()

  add_library("${target}" SHARED ${FLAGDNN_BACKEND_SOURCES})
  add_library("FlagDNN::backend_${backend_name}" ALIAS "${target}")
  target_compile_features("${target}" PRIVATE cxx_std_20)
  target_include_directories("${target}" PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/generated/include
    ${FLAGDNN_BACKEND_INCLUDE_DIRECTORIES})
  if(FLAGDNN_BACKEND_LINK_LIBRARIES)
    target_link_libraries("${target}" PRIVATE
      ${FLAGDNN_BACKEND_LINK_LIBRARIES})
  endif()
  flagdnn_enable_warnings("${target}")

  if(UNIX AND NOT APPLE)
    if(FLAGDNN_BACKEND_ABI_VERSION EQUAL 2)
      set(_flagdnn_backend_version_script
        "${PROJECT_SOURCE_DIR}/cmake/flagdnn_backend.map")
    else()
      set(_flagdnn_backend_version_script
        "${PROJECT_SOURCE_DIR}/cmake/flagdnn_backend_v3.map")
    endif()
    target_link_options("${target}" PRIVATE
      "LINKER:--version-script=${_flagdnn_backend_version_script}")
    set_property(TARGET "${target}" APPEND PROPERTY LINK_DEPENDS
      "${_flagdnn_backend_version_script}")
  endif()
  set_target_properties("${target}" PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION "${FLAGDNN_BACKEND_ABI_VERSION}.0.0"
    SOVERSION "${FLAGDNN_BACKEND_ABI_VERSION}")

  if(FLAGDNN_BACKEND_INSTALL)
    install(TARGETS "${target}"
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endfunction()
