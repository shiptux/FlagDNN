/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/context.hpp"
#include "backends/ascend/error.hpp"

#include <Python.h>
#include <triton_jit/kernel_metadata.h>

#include <acl/acl_rt.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

/* context.cpp verifies the origin of this public LTJ symbol in production.
 * The pure domain-state harness deliberately does not load LTJ/Torch, so a
 * never-called definition keeps that host-only seam independent. */
namespace triton_jit {
NpuKernelMetadata load_npu_metadata(const std::string&,
                                    const std::string&) {
  return {};
}
}  // namespace triton_jit

namespace {

using flagdnn::ascend::AscendError;
using flagdnn::ascend::ContextGuard;
using flagdnn::ascend::detail::AclRuntimeApi;
using flagdnn::ascend::detail::DomainBinding;
using flagdnn::ascend::detail::DomainCoordinator;
using flagdnn::ascend::detail::DomainInitialization;
using flagdnn::ascend::detail::DomainPhase;
using flagdnn::ascend::detail::InitializationDisposition;

[[nodiscard]] aclrtContext fake_context(std::uintptr_t value) {
  return reinterpret_cast<aclrtContext>(value);
}

void check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_unique_loaded_python_host_identity() {
  check(flagdnn::ascend::detail::loaded_python_object_count_for_test() == 1,
        "domain-test host did not expose exactly one canonical libpython DSO");
  flagdnn::ascend::detail::verify_pinned_python_files_for_test();
}

struct Outcome {
  std::shared_ptr<const DomainBinding> binding;
  flagdnnBackendResult_t error = FLAGDNN_BACKEND_RESULT_SUCCESS;
  std::string message;
};

void acquire_into(DomainCoordinator& domain,
                  std::int32_t device,
                  const DomainCoordinator::Initializer& initializer,
                  Outcome* outcome) {
  try {
    outcome->binding = domain.acquire(device, initializer);
  } catch (const AscendError& error) {
    outcome->error = error.result();
    outcome->message = error.what();
  } catch (const std::exception& error) {
    outcome->error = FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR;
    outcome->message = error.what();
  }
}

[[nodiscard]] DomainBinding binding_for(std::int32_t device) {
  DomainBinding binding;
  binding.device_ordinal = device;
  binding.default_context = fake_context(0x1000U +
                                         static_cast<std::uintptr_t>(device));
  binding.ai_core_count = 24;
  binding.target_fingerprint = "mock_ascend_" + std::to_string(device);
  binding.codegen_arch = "Ascend910B1";
  binding.runtime_identity = "mock-runtime";
  binding.development_mode = true;
  binding.production_cache_root = "/mock/private/cache";
  binding.configuration_identity = "mock-configuration";
  return binding;
}

void test_atomic_concurrent_bind() {
  DomainCoordinator domain;
  std::atomic<int> initializer_calls{0};
  std::promise<void> initializer_entered;
  std::promise<void> release_promise;
  std::shared_future<void> release_initializer =
      release_promise.get_future().share();

  const DomainCoordinator::Initializer initializer = [&](std::int32_t device) {
    ++initializer_calls;
    initializer_entered.set_value();
    release_initializer.wait();
    return DomainInitialization::bound(binding_for(device));
  };

  std::vector<Outcome> outcomes(8);
  std::vector<std::thread> threads;
  threads.emplace_back(acquire_into,
                       std::ref(domain),
                       0,
                       std::cref(initializer),
                       &outcomes[0]);
  initializer_entered.get_future().wait();
  check(domain.snapshot().phase == DomainPhase::kInitializing,
        "domain did not publish INITIALIZING");
  check(domain.snapshot().requested_device == 0,
        "initializer device was partially or incorrectly published");
  for (std::size_t index = 1; index + 1 < outcomes.size(); ++index) {
    threads.emplace_back(acquire_into,
                         std::ref(domain),
                         0,
                         std::cref(initializer),
                         &outcomes[index]);
  }
  threads.emplace_back(acquire_into,
                       std::ref(domain),
                       1,
                       std::cref(initializer),
                       &outcomes.back());
  release_promise.set_value();
  for (std::thread& thread : threads) {
    thread.join();
  }

  check(initializer_calls.load() == 1,
        "more than one first-use initializer ran");
  const auto first = outcomes.front().binding;
  check(first != nullptr, "initializer did not commit a binding");
  check(first->development_mode && first->ai_core_count == 24 &&
            first->production_cache_root == "/mock/private/cache" &&
            first->configuration_identity == "mock-configuration",
        "domain binding lost its frozen development configuration");
  for (std::size_t index = 0; index + 1 < outcomes.size(); ++index) {
    check(outcomes[index].error == FLAGDNN_BACKEND_RESULT_SUCCESS,
          "same-domain waiter failed");
    check(outcomes[index].binding == first,
          "same-domain waiters observed different bindings");
  }
  check(outcomes.back().error == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED &&
            outcomes.back().binding == nullptr,
        "concurrent different-domain waiter was accepted");
  check(domain.snapshot().phase == DomainPhase::kBound,
        "domain did not atomically commit BOUND");
  check(domain.bound_device_ordinal(first->default_context) == 0,
        "bound context did not resolve to its device ordinal");
  try {
    (void)domain.bound_device_ordinal(fake_context(0xdeadU));
    throw std::runtime_error("foreign context resolved in the bound domain");
  } catch (const AscendError& error) {
    check(error.result() == FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
          "foreign bound-context lookup returned the wrong result");
  }

  int unexpected_calls = 0;
  const DomainCoordinator::Initializer must_not_run =
      [&](std::int32_t device) {
        ++unexpected_calls;
        return DomainInitialization::bound(binding_for(device));
      };
  try {
    (void)domain.acquire(1, must_not_run);
    throw std::runtime_error("different device was accepted");
  } catch (const AscendError& error) {
    check(error.result() == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
          "different device returned the wrong result");
  }
  check(unexpected_calls == 0,
        "different device started a second initializer");
  check(domain.snapshot().phase == DomainPhase::kBound,
        "different device changed the bound domain");
}

void test_retryable_wakes_to_unbound() {
  DomainCoordinator domain;
  std::atomic<int> initializer_calls{0};
  std::promise<void> first_entered;
  std::promise<void> release_first;
  std::shared_future<void> release = release_first.get_future().share();
  const DomainCoordinator::Initializer initializer = [&](std::int32_t device) {
    const int call = ++initializer_calls;
    if (call == 1) {
      first_entered.set_value();
      release.wait();
      return DomainInitialization::retryable(
          FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
          "mock caller context is not ready");
    }
    return DomainInitialization::bound(binding_for(device));
  };

  Outcome first;
  Outcome waiter;
  std::thread first_thread(acquire_into,
                           std::ref(domain),
                           0,
                           std::cref(initializer),
                           &first);
  first_entered.get_future().wait();
  std::thread waiter_thread(acquire_into,
                            std::ref(domain),
                            0,
                            std::cref(initializer),
                            &waiter);
  release_first.set_value();
  first_thread.join();
  waiter_thread.join();

  check(first.error == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "retryable initializer did not return its error");
  check(waiter.binding != nullptr,
        "waiter did not re-evaluate UNBOUND after notification");
  check(initializer_calls.load() == 2,
        "retryable transition did not permit exactly one new initializer");
  check(domain.snapshot().phase == DomainPhase::kBound,
        "retry path did not eventually bind");
}

void test_terminal_wakes_and_latches() {
  DomainCoordinator domain;
  std::atomic<int> initializer_calls{0};
  std::promise<void> first_entered;
  std::promise<void> release_first;
  std::shared_future<void> release = release_first.get_future().share();
  const DomainCoordinator::Initializer initializer = [&](std::int32_t) {
    ++initializer_calls;
    first_entered.set_value();
    release.wait();
    return DomainInitialization::terminal(
        FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR, "mock terminal initialization");
  };

  Outcome first;
  Outcome waiter;
  std::thread first_thread(acquire_into,
                           std::ref(domain),
                           0,
                           std::cref(initializer),
                           &first);
  first_entered.get_future().wait();
  std::thread waiter_thread(acquire_into,
                            std::ref(domain),
                            0,
                            std::cref(initializer),
                            &waiter);
  release_first.set_value();
  first_thread.join();
  waiter_thread.join();

  check(initializer_calls.load() == 1,
        "terminal transition allowed a second initializer");
  check(first.error == FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR &&
            waiter.error == FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
        "terminal transition did not wake all callers with failure");
  check(domain.snapshot().phase == DomainPhase::kFailed &&
            domain.terminal_failure_latched(),
        "terminal failure was not latched");

  bool reran = false;
  try {
    (void)domain.acquire(0, [&](std::int32_t device) {
      reran = true;
      return DomainInitialization::bound(binding_for(device));
    });
    throw std::runtime_error("terminal domain accepted a new handle");
  } catch (const AscendError& error) {
    check(error.result() == FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
          "latched failure changed its result");
  }
  check(!reran, "terminal domain reran its initializer");
}

void test_initializer_exception_notifies() {
  DomainCoordinator domain;
  std::atomic<int> initializer_calls{0};
  std::promise<void> first_entered;
  std::promise<void> release_first;
  std::shared_future<void> release = release_first.get_future().share();
  const DomainCoordinator::Initializer initializer =
      [&](std::int32_t) -> DomainInitialization {
    ++initializer_calls;
    first_entered.set_value();
    release.wait();
    throw std::runtime_error("mock initializer exception");
  };

  Outcome first;
  Outcome waiter;
  std::thread first_thread(acquire_into,
                           std::ref(domain),
                           0,
                           std::cref(initializer),
                           &first);
  first_entered.get_future().wait();
  std::thread waiter_thread(acquire_into,
                            std::ref(domain),
                            0,
                            std::cref(initializer),
                            &waiter);
  release_first.set_value();
  first_thread.join();
  waiter_thread.join();

  check(initializer_calls.load() == 1,
        "initializer exception allowed another initializer");
  check(first.error == FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR &&
            waiter.error == FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
        "initializer exception did not notify both callers");
  check(domain.snapshot().phase == DomainPhase::kFailed &&
            domain.terminal_failure_latched(),
        "initializer exception did not commit FAILED");
}

struct FakeAclState {
  aclrtContext current = nullptr;
  aclError get_result = ACL_SUCCESS;
  aclError readback_result = ACL_SUCCESS;
  aclError set_result = ACL_SUCCESS;
  aclError device_result = ACL_SUCCESS;
  std::int32_t ai_core_result = 0;
  std::int32_t device = 0;
  std::uint32_t ai_core_count = 24;
  bool set_updates_current = true;
  int get_calls = 0;
  int set_calls = 0;
  int device_calls = 0;
  int ai_core_calls = 0;
};

FakeAclState* fake_acl_state = nullptr;

aclError fake_get_current(aclrtContext* context) {
  ++fake_acl_state->get_calls;
  const aclError result = fake_acl_state->get_calls == 1
                              ? fake_acl_state->get_result
                              : fake_acl_state->readback_result;
  if (result == ACL_SUCCESS) {
    *context = fake_acl_state->current;
  }
  return result;
}

aclError fake_set_current(aclrtContext context) {
  ++fake_acl_state->set_calls;
  if (fake_acl_state->set_result == ACL_SUCCESS &&
      fake_acl_state->set_updates_current) {
    fake_acl_state->current = context;
  }
  return fake_acl_state->set_result;
}

aclError fake_get_device(std::int32_t* device) {
  ++fake_acl_state->device_calls;
  if (fake_acl_state->device_result == ACL_SUCCESS) {
    *device = fake_acl_state->device;
  }
  return fake_acl_state->device_result;
}

std::int32_t fake_get_ai_core_count(std::uint32_t* ai_core_count) {
  ++fake_acl_state->ai_core_calls;
  if (fake_acl_state->ai_core_result == 0) {
    *ai_core_count = fake_acl_state->ai_core_count;
  }
  return fake_acl_state->ai_core_result;
}

std::int32_t unexpected_get_ai_core_count(std::uint32_t*) {
  throw std::runtime_error("preflight called rtGetAiCoreCount");
}

aclError unexpected_get_device(std::int32_t*) {
  throw std::runtime_error("production preflight called aclrtGetDevice");
}

aclError fake_get_device_zero(std::int32_t* device) {
  return fake_get_device(device);
}

const char* unexpected_get_soc() {
  throw std::runtime_error("production preflight called aclrtGetSocName");
}

const char* unsupported_safe_soc() {
  return "Ascend950A1";
}

const char* supported_soc() {
  return "Ascend910B1";
}

aclError unexpected_get_version(char*, char*) {
  throw std::runtime_error("production preflight called version query");
}

aclError unexpected_get_version_number(char*, std::int32_t*) {
  throw std::runtime_error("production preflight called version query");
}

void test_context_guard_policy() {
  const aclrtContext expected = fake_context(0x2000U);
  const aclrtContext different = fake_context(0x3000U);
  const AclRuntimeApi api = {
      &fake_get_current,
      &fake_set_current,
      &fake_get_device,
      nullptr,
      nullptr,
      nullptr,
      nullptr};

  {
    DomainCoordinator domain;
    FakeAclState state;
    state.current = expected;
    fake_acl_state = &state;
    ContextGuard guard(expected, 0, domain, api);
    check(state.get_calls == 1 && state.set_calls == 0,
          "same current context was unnecessarily changed");
  }
  {
    DomainCoordinator domain;
    FakeAclState state;
    fake_acl_state = &state;
    {
      ContextGuard guard(expected, 0, domain, api);
    }
    check(state.current == expected && state.set_calls == 1 &&
              state.get_calls == 2 && state.device_calls == 1,
          "null current context was not restored and read back exactly");
    check(state.current == expected,
          "ContextGuard destructor incorrectly cleared/restored null current");
  }
  {
    DomainCoordinator domain;
    FakeAclState state;
    state.current = different;
    fake_acl_state = &state;
    try {
      ContextGuard guard(expected, 0, domain, api);
      throw std::runtime_error("different non-null context was accepted");
    } catch (const AscendError& error) {
      check(error.result() == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
            "different context returned the wrong result");
    }
    check(state.current == different && state.set_calls == 0,
          "different context was silently switched");
  }
  {
    DomainCoordinator domain;
    FakeAclState state;
    state.set_result = 107002;
    fake_acl_state = &state;
    try {
      ContextGuard guard(expected, 0, domain, api);
      throw std::runtime_error("failed context restore was accepted");
    } catch (const AscendError& error) {
      check(error.result() == FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
            "failed context restore returned the wrong result");
    }
    check(domain.terminal_failure_latched() &&
              domain.snapshot().phase == DomainPhase::kFailed,
          "failed context restore did not latch the domain terminally");
  }
  {
    DomainCoordinator domain;
    FakeAclState state;
    state.set_updates_current = false;
    fake_acl_state = &state;
    try {
      ContextGuard guard(expected, 0, domain, api);
      throw std::runtime_error("context restore without readback was accepted");
    } catch (const AscendError& error) {
      check(error.result() == FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
            "context readback mismatch returned the wrong result");
    }
    check(domain.terminal_failure_latched() && state.get_calls == 2,
          "context readback mismatch did not latch the domain");
  }
  {
    DomainCoordinator domain;
    FakeAclState state;
    state.device = 1;
    fake_acl_state = &state;
    try {
      ContextGuard guard(expected, 0, domain, api);
      throw std::runtime_error("restored wrong-device context was accepted");
    } catch (const AscendError& error) {
      check(error.result() == FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
            "restored wrong device returned the wrong result");
    }
    check(domain.terminal_failure_latched() && state.device_calls == 1,
          "restored wrong device did not latch the domain");
  }
  {
    DomainCoordinator domain;
    FakeAclState state;
    state.get_result = ACL_ERROR_RT_CONTEXT_NULL;
    fake_acl_state = &state;
    ContextGuard guard(expected, 0, domain, api);
    check(state.current == expected && state.set_calls == 1 &&
              state.get_calls == 2 && state.device_calls == 1,
          "ACL_ERROR_RT_CONTEXT_NULL was not restored and read back");
    check(!domain.terminal_failure_latched(),
          "recoverable null context terminalized the domain");
  }
  {
    DomainCoordinator domain;
    FakeAclState state;
    state.get_result = ACL_ERROR_RT_PARAM_INVALID;
    fake_acl_state = &state;
    try {
      ContextGuard guard(expected, 0, domain, api);
      throw std::runtime_error("failed context query was accepted");
    } catch (const AscendError& error) {
      check(error.result() == FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
            "failed context query returned the wrong result");
    }
    check(domain.terminal_failure_latched() &&
              domain.snapshot().phase == DomainPhase::kFailed,
          "failed context query did not latch the domain terminally");
  }
  fake_acl_state = nullptr;
}

class ScopedEnvironment {
 public:
  explicit ScopedEnvironment(const char* name) : name_(name) {
    const char* value = std::getenv(name);
    if (value != nullptr) {
      old_value_ = value;
    }
  }

  ~ScopedEnvironment() {
    if (old_value_.has_value()) {
      (void)::setenv(name_.c_str(), old_value_->c_str(), 1);
    } else {
      (void)::unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> old_value_;
};

void test_unsupported_safe_soc_terminalizes() {
  ScopedEnvironment containment("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT");
  (void)::setenv("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT", "development", 1);

  FakeAclState state;
  state.current = fake_context(0x4000U);
  fake_acl_state = &state;
  const AclRuntimeApi api = {
      &fake_get_current,
      &fake_set_current,
      &fake_get_device_zero,
      &unexpected_get_ai_core_count,
      &unsupported_safe_soc,
      &unexpected_get_version,
      &unexpected_get_version_number,
  };
  DomainCoordinator domain;
  int initializer_calls = 0;
  const DomainCoordinator::Initializer initializer = [&](std::int32_t device) {
    ++initializer_calls;
    return flagdnn::ascend::detail::initialize_domain_with_api(device, api);
  };

  try {
    (void)domain.acquire(0, initializer);
    throw std::runtime_error("unsupported safe SoC was accepted");
  } catch (const AscendError& error) {
    check(error.result() == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED &&
              std::string(error.what()).find("unsupported") !=
                  std::string::npos,
          "unsupported safe SoC returned the wrong terminal failure");
  }
  check(initializer_calls == 1 &&
            domain.snapshot().phase == DomainPhase::kFailed &&
            domain.terminal_failure_latched(),
        "unsupported safe SoC did not terminalize the domain");

  try {
    (void)domain.acquire(0, initializer);
    throw std::runtime_error("terminal unsupported-SoC domain was reopened");
  } catch (const AscendError& error) {
    check(error.result() == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
          "unsupported-SoC terminal latch changed its result");
  }
  check(initializer_calls == 1,
        "unsupported-SoC terminal latch reran the initializer");
  fake_acl_state = nullptr;
}

void test_ai_core_capability_probe() {
  ScopedEnvironment containment("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT");
  (void)::setenv("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT", "development", 1);

  FakeAclState state;
  state.current = fake_context(0x5000U);
  fake_acl_state = &state;
  const AclRuntimeApi api = {
      &fake_get_current,
      &fake_set_current,
      &fake_get_device_zero,
      &fake_get_ai_core_count,
      &supported_soc,
      &unexpected_get_version,
      &unexpected_get_version_number,
  };

  state.ai_core_result = 107002;
  DomainInitialization result =
      flagdnn::ascend::detail::initialize_domain_with_api(0, api);
  check(result.disposition == InitializationDisposition::kRetryableFailure &&
            result.result == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED &&
            state.ai_core_calls == 1,
        "rtGetAiCoreCount failure did not stay retryable");

  state.get_calls = 0;
  state.device_calls = 0;
  state.ai_core_calls = 0;
  state.ai_core_result = 0;
  state.ai_core_count = 0;
  result = flagdnn::ascend::detail::initialize_domain_with_api(0, api);
  check(result.disposition == InitializationDisposition::kTerminalFailure &&
            result.result == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED &&
            state.ai_core_calls == 1,
        "zero AI core capability was accepted");

  AclRuntimeApi incomplete = api;
  incomplete.get_ai_core_count = nullptr;
  result =
      flagdnn::ascend::detail::initialize_domain_with_api(0, incomplete);
  check(result.disposition == InitializationDisposition::kTerminalFailure &&
            result.result == FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
        "incomplete AI core probe API was accepted");
  fake_acl_state = nullptr;
}

void test_parallel_blocks_policy_conflict() {
  ScopedEnvironment containment("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT");
  ScopedEnvironment parallel("TRITON_ALL_BLOCKS_PARALLEL");
  (void)::setenv("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT", "development", 1);
  (void)::setenv("TRITON_ALL_BLOCKS_PARALLEL", "true", 1);

  FakeAclState state;
  state.current = fake_context(0x6000U);
  fake_acl_state = &state;
  const AclRuntimeApi api = {
      &fake_get_current,
      &fake_set_current,
      &fake_get_device_zero,
      &fake_get_ai_core_count,
      &supported_soc,
      &unexpected_get_version,
      &unexpected_get_version_number,
  };
  const DomainInitialization result =
      flagdnn::ascend::detail::initialize_domain_with_api(0, api);
  check(result.disposition == InitializationDisposition::kTerminalFailure &&
            result.result == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED &&
            result.message.find("TRITON_ALL_BLOCKS_PARALLEL") !=
                std::string::npos,
        "conflicting all-blocks-parallel mode was accepted");
  fake_acl_state = nullptr;
}

void test_local_default_and_hardened_preflight() {
  ScopedEnvironment containment("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT");
  (void)::unsetenv("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT");
  check(Py_IsInitialized() == 0,
        "domain test parent unexpectedly started with CPython initialized");
  FakeAclState state;
  fake_acl_state = &state;
  const AclRuntimeApi api = {
      &fake_get_current,
      &fake_set_current,
      &unexpected_get_device,
      &unexpected_get_ai_core_count,
      &unexpected_get_soc,
      &unexpected_get_version,
      &unexpected_get_version_number,
  };
  DomainInitialization result =
      flagdnn::ascend::detail::initialize_domain_with_api(0, api);
  check(result.disposition == InitializationDisposition::kRetryableFailure &&
            result.result == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "default trusted-local context probe returned the wrong result");
  check(state.get_calls == 1 && state.set_calls == 0,
        "default trusted-local mode did not enter the ACL context probe");
  check(Py_IsInitialized() == 0,
        "trusted-local context precondition failure initialized CPython");

  (void)::setenv("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT", "trusted-local", 1);
  state.get_calls = 0;
  result = flagdnn::ascend::detail::initialize_domain_with_api(0, api);
  check(result.disposition == InitializationDisposition::kRetryableFailure &&
            result.result == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED &&
            state.get_calls == 1,
        "explicit trusted-local mode did not enter the ACL context probe");

  (void)::setenv("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT", "development", 1);
  state.get_calls = 0;
  state.get_result = 107002;
  result =
      flagdnn::ascend::detail::initialize_domain_with_api(0, api);
  check(result.disposition == InitializationDisposition::kRetryableFailure &&
            result.result == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "explicit development context probe returned the wrong result");
  check(state.get_calls == 1,
        "explicit development mode did not enter the ACL context probe");
  check(Py_IsInitialized() == 0,
        "development context precondition failure initialized CPython");

  for (const char* hardened : {"production", "hardened"}) {
    (void)::setenv("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT", hardened, 1);
    state.get_calls = 0;
    result = flagdnn::ascend::detail::initialize_domain_with_api(0, api);
    check(result.disposition == InitializationDisposition::kRetryableFailure &&
              result.result == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
          "incomplete hardened containment did not fail fast");
    check(state.get_calls == 0 && state.set_calls == 0,
          "hardened preflight touched the ACL context before rejection");
    check(Py_IsInitialized() == 0,
          "hardened preflight initialized CPython before rejection");
    check(result.message.find("production") != std::string::npos ||
              result.message.find("cgroup") != std::string::npos ||
              result.message.find("hardened") != std::string::npos,
          "hardened failure lacks a containment diagnostic");
  }

  (void)::setenv("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT", "unsupported", 1);
  state.get_calls = 0;
  result = flagdnn::ascend::detail::initialize_domain_with_api(0, api);
  check(result.disposition == InitializationDisposition::kRetryableFailure &&
            result.result == FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED &&
            state.get_calls == 0 &&
            result.message.find("trusted-local") != std::string::npos,
        "invalid containment mode was not rejected before ACL probing");
  fake_acl_state = nullptr;
}

}  // namespace

int main() {
  try {
    test_unique_loaded_python_host_identity();
    test_atomic_concurrent_bind();
    test_retryable_wakes_to_unbound();
    test_terminal_wakes_and_latches();
    test_initializer_exception_notifies();
    test_context_guard_policy();
    test_unsupported_safe_soc_terminalizes();
    test_ai_core_capability_probe();
    test_parallel_blocks_policy_conflict();
    test_local_default_and_hardened_preflight();
    std::cout << "Ascend domain/context contract checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Ascend domain/context contract check failed: "
              << error.what() << '\n';
    return 1;
  }
}
