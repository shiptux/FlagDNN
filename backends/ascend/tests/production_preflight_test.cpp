/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/backend_api.h"

#include <dlfcn.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    check(argc == 3, "expected plugin path and cache sentinel path");
    const std::filesystem::path cache_sentinel(argv[2]);
    check(!std::filesystem::exists(cache_sentinel),
          "cache sentinel unexpectedly exists before preflight");
    check(::setenv("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT", "production", 1) ==
              0,
          "cannot select production containment");
    check(::setenv("TRITON_CACHE_DIR", cache_sentinel.c_str(), 1) == 0,
          "cannot install cache sentinel");

    void* library = ::dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      const char* error = ::dlerror();
      throw std::runtime_error(
          "cannot load Ascend plugin: " +
          std::string(error == nullptr ? "unknown dlopen error" : error));
    }
    const auto get_api = reinterpret_cast<flagdnnBackendGetApiV3Function>(
        ::dlsym(library, FLAGDNN_BACKEND_GET_API_V3_SYMBOL));
    check(get_api != nullptr, "Ascend plugin has no backend ABI v3 getter");
    const flagdnnBackendApiV3* api = get_api();
    check(api != nullptr &&
              api->abi_version == FLAGDNN_BACKEND_ABI_VERSION_V3 &&
              api->execution_contract_version ==
                  FLAGDNN_BACKEND_EXECUTION_CONTRACT_VERSION,
          "Ascend plugin returned an incompatible backend API");

    void* context = reinterpret_cast<void*>(1);
    const flagdnnBackendResult_t result = api->create_context(0, &context);
    check(result == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
          "incomplete production infrastructure was not rejected");
    check(context == nullptr,
          "failed production create_context published a partial context");
    const char* diagnostic = api->get_last_error();
    check(diagnostic != nullptr && diagnostic[0] != '\0',
          "production rejection has no diagnostic");
    const std::string message(diagnostic);
    check(message.find("production") != std::string::npos ||
              message.find("cgroup") != std::string::npos ||
              message.find("containment") != std::string::npos,
          "production rejection does not identify the resource gate");
    check(!std::filesystem::exists(cache_sentinel),
          "production preflight created or mutated the cache sentinel");

    using PyIsInitialized = int (*)();
    const auto py_is_initialized = reinterpret_cast<PyIsInitialized>(
        ::dlsym(library, "Py_IsInitialized"));
    check(py_is_initialized != nullptr,
          "plugin does not expose its matching libpython dependency");
    check(py_is_initialized() == 0,
          "production preflight initialized CPython before resource gates");
    std::cout << "Ascend production preflight fail-fast check passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Ascend production preflight check failed: " << error.what()
              << '\n';
    return 1;
  }
}
