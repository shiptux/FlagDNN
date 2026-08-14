cmake_minimum_required(VERSION 3.23)

project(FlagDNNInstalledConsumer LANGUAGES C CXX)

include(CTest)

find_package(FlagDNN 0.1 CONFIG REQUIRED)

option(FLAGDNN_INSTALLED_NVIDIA_ADD
  "Build and run an installed-SDK NVIDIA Add integration consumer" OFF)
option(FLAGDNN_INSTALLED_HYGON_ADD
  "Build and run an installed-SDK Hygon Add integration consumer" OFF)
set(FLAGDNN_INSTALLED_CODEGEN_PYTHON "" CACHE FILEPATH
  "Python interpreter containing Triton for an installed GPU test")
set(FLAGDNN_INSTALLED_HYGON_SDK_ROOT "" CACHE PATH
  "Isolated FlagDNN SDK root for the installed Hygon contract")
set(FLAGDNN_INSTALLED_HYGON_JIT_LIBRARY "" CACHE FILEPATH
  "Hygon-private installed libtriton_jit image")
set(FLAGDNN_INSTALLED_HYGON_JIT_SCRIPT_DIR "" CACHE PATH
  "Private script directory resolved by the installed HCU JIT image")

add_executable(flagdnn_installed_c_consumer main.c)
target_link_libraries(flagdnn_installed_c_consumer PRIVATE FlagDNN::flagdnn)
set_target_properties(flagdnn_installed_c_consumer PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES)
add_test(NAME installed.c COMMAND flagdnn_installed_c_consumer)

add_executable(flagdnn_installed_cpp_consumer main.cpp)
target_compile_features(flagdnn_installed_cpp_consumer PRIVATE cxx_std_20)
target_link_libraries(flagdnn_installed_cpp_consumer PRIVATE FlagDNN::flagdnn)
add_test(NAME installed.cpp COMMAND flagdnn_installed_cpp_consumer)

if(FLAGDNN_INSTALLED_NVIDIA_ADD)
  if(NOT FLAGDNN_INSTALLED_CODEGEN_PYTHON OR
     NOT EXISTS "${FLAGDNN_INSTALLED_CODEGEN_PYTHON}")
    message(FATAL_ERROR
      "FLAGDNN_INSTALLED_CODEGEN_PYTHON must name the test Python containing Triton")
  endif()
  find_package(CUDAToolkit REQUIRED)
  add_executable(flagdnn_installed_nvidia_add nvidia_add.cpp)
  target_compile_features(flagdnn_installed_nvidia_add PRIVATE cxx_std_20)
  target_link_libraries(flagdnn_installed_nvidia_add PRIVATE
    FlagDNN::flagdnn CUDA::cuda_driver)

  add_test(NAME installed.nvidia_add COMMAND flagdnn_installed_nvidia_add)
  set_tests_properties(installed.nvidia_add PROPERTIES
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:FlagDNN::flagdnn>;FLAGDNN_COMPILER_EXECUTABLE=${FLAGDNN_INSTALLED_CODEGEN_PYTHON}"
    TIMEOUT 600)
endif()

if(FLAGDNN_INSTALLED_HYGON_ADD)
  if(NOT FLAGDNN_INSTALLED_CODEGEN_PYTHON OR
     NOT EXISTS "${FLAGDNN_INSTALLED_CODEGEN_PYTHON}")
    message(FATAL_ERROR
      "FLAGDNN_INSTALLED_CODEGEN_PYTHON must name the test Python containing Triton")
  endif()
  foreach(_flagdnn_installed_hygon_resource IN ITEMS
      FLAGDNN_INSTALLED_HYGON_SDK_ROOT
      FLAGDNN_INSTALLED_HYGON_JIT_LIBRARY
      FLAGDNN_INSTALLED_HYGON_JIT_SCRIPT_DIR)
    if("${${_flagdnn_installed_hygon_resource}}" STREQUAL "" OR
       NOT EXISTS "${${_flagdnn_installed_hygon_resource}}")
      message(FATAL_ERROR
        "${_flagdnn_installed_hygon_resource} must name an installed SDK resource")
    endif()
  endforeach()
  find_package(hip CONFIG REQUIRED)
  if(NOT TARGET hip::host)
    message(FATAL_ERROR "DTK HIP package did not export hip::host")
  endif()
  find_path(FLAGDNN_INSTALLED_HIPDNN_INCLUDE_DIR
    NAMES hipdnn.h
    HINTS "$ENV{HIPDNN_ROOT}" "$ENV{DTK_ROOT}" /opt/dtk/hipdnn /opt/dtk
    PATH_SUFFIXES include hipdnn/include
    REQUIRED)
  find_library(FLAGDNN_INSTALLED_HIPDNN_LIBRARY
    NAMES hipdnn
    HINTS "$ENV{HIPDNN_ROOT}" "$ENV{DTK_ROOT}" /opt/dtk/hipdnn /opt/dtk
    PATH_SUFFIXES lib lib64
    REQUIRED)

  add_executable(flagdnn_installed_hygon_add hygon_add.cpp)
  target_compile_features(flagdnn_installed_hygon_add PRIVATE cxx_std_20)
  target_include_directories(flagdnn_installed_hygon_add PRIVATE
    "${FLAGDNN_INSTALLED_HIPDNN_INCLUDE_DIR}")
  target_link_libraries(flagdnn_installed_hygon_add PRIVATE
    FlagDNN::flagdnn
    hip::host
    "${FLAGDNN_INSTALLED_HIPDNN_LIBRARY}"
    "${CMAKE_DL_LIBS}")
  get_filename_component(_flagdnn_installed_hipdnn_library_directory
    "${FLAGDNN_INSTALLED_HIPDNN_LIBRARY}" DIRECTORY)
  set_target_properties(flagdnn_installed_hygon_add PROPERTIES
    BUILD_RPATH "${_flagdnn_installed_hipdnn_library_directory}")

  add_test(NAME installed.hygon_add COMMAND flagdnn_installed_hygon_add)
  set_tests_properties(installed.hygon_add PROPERTIES
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:FlagDNN::flagdnn>;FLAGDNN_COMPILER_EXECUTABLE=${FLAGDNN_INSTALLED_CODEGEN_PYTHON};FLAGDNN_INSTALLED_EXPECTED_SDK=${FLAGDNN_INSTALLED_HYGON_SDK_ROOT};FLAGDNN_INSTALLED_EXPECTED_JIT=${FLAGDNN_INSTALLED_HYGON_JIT_LIBRARY};FLAGDNN_INSTALLED_EXPECTED_JIT_SCRIPT_DIR=${FLAGDNN_INSTALLED_HYGON_JIT_SCRIPT_DIR}"
    TIMEOUT 1200)
  unset(_flagdnn_installed_hipdnn_library_directory)
endif()
