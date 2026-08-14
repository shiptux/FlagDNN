/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/reduction.hpp"
#include "tensor_runner_support.hpp"

#include <cstdlib>
#include <functional>
#include <span>

namespace flagdnn::testing {

int run_reduction_functional_test(int argc, char **argv,
                                  std::span<const ReductionTestCase> cases) {
  namespace support = hygon_functional::tensor;
  namespace hv = validation::hygon;
  return support::run_suite(
      argc, argv, "reduction", "FLAGDNN_REDUCTION_FUNCTIONAL",
      [&](const std::function<flagdnn::Handle &()> &get_handle,
          hv::Stream &stream, std::size_t &matched, std::size_t &executed,
          std::size_t &skipped) {
        const char *filter = std::getenv("FLAGDNN_REDUCTION_CASE");
        for (const ReductionTestCase &test_case : cases) {
          if (filter != nullptr &&
              test_case.name.find(filter) == std::string::npos) {
            continue;
          }
          ++matched;
          validate_reduction_case(test_case);
          const support::CaseResult result =
              support::run_reduction_case(test_case, get_handle, stream);
          result == support::CaseResult::kExecuted ? ++executed : ++skipped;
        }
      });
}

} // namespace flagdnn::testing
