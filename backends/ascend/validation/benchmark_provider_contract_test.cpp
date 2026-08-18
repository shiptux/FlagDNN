/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/benchmark/aclnn_provider.hpp"

#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

flagdnn::benchmarking::BenchmarkCase valid_add_case() {
  namespace benchmarking = flagdnn::benchmarking;
  benchmarking::BenchmarkCase result;
  result.name = "add_exact_reference_contract";
  result.operation = benchmarking::Operation::kAdd;
  result.tensors = {
      benchmarking::tensor(1, {2, 3}),
      benchmarking::tensor(2, {2, 3}),
      benchmarking::tensor(3, {2, 3}),
  };
  return result;
}

}  // namespace

int main() {
  namespace benchmarking = flagdnn::benchmarking;
  benchmarking::AclnnProvider provider;

  const benchmarking::ProviderCapability supported =
      provider.capability(valid_add_case());
  require(supported.supported,
          "valid Add must have an exact ACLNN benchmark mapping");

  benchmarking::BenchmarkCase no_mapping;
  no_mapping.name = "graph_without_exact_aclnn_mapping";
  no_mapping.operation = benchmarking::Operation::kGraph;
  const benchmarking::ProviderCapability unsupported =
      provider.capability(no_mapping);
  require(!unsupported.supported && !unsupported.reason.empty(),
          "an explicit missing ACLNN mapping must be unsupported");

  benchmarking::BenchmarkCase malformed = valid_add_case();
  malformed.name = "malformed_add";
  malformed.tensors.pop_back();
  bool rejected = false;
  try {
    (void)provider.capability(malformed);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected,
          "malformed benchmark cases must fail instead of becoming SKIP");
  return 0;
}
