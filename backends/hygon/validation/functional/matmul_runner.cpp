/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/matmul.hpp"
#include "tensor_runner_support.hpp"

#include <cstdlib>
#include <functional>
#include <span>
#include <stdexcept>

namespace flagdnn::testing {

int run_matmul_functional_test(int argc, char **argv,
                               std::span<const MatmulTestCase> cases) {
  namespace support = hygon_functional::tensor;
  namespace hv = validation::hygon;
  return support::run_suite(
      argc, argv, "matmul", "FLAGDNN_MATMUL_FUNCTIONAL",
      [&](const std::function<flagdnn::Handle &()> &, hv::Stream &,
          std::size_t &matched, std::size_t &executed, std::size_t &skipped) {
        const char *filter = std::getenv("FLAGDNN_MATMUL_CASE");
        for (const MatmulTestCase &test_case : cases) {
          if (filter != nullptr &&
              test_case.name.find(filter) == std::string::npos) {
            continue;
          }
          ++matched;
          validate_matmul_case(test_case);
          const hv::HipdnnTensorOperation operation =
              support::matmul_operation();
          const std::vector<hv::ReferenceTensor> tensors =
              support::matmul_reference_tensors(test_case);
          const hv::HipdnnCapability capability =
              hv::hipdnn_tensor_capability(operation, tensors);
          hv::require_valid_hipdnn_adapter_contract(capability, "matmul");
          if (capability.supported) {
            throw std::runtime_error(
                "hipDNN MatMul capability unexpectedly became supported");
          }
          support::emit_skip("matmul", test_case.name, capability.reason,
                             tensors);
          ++skipped;
        }
        (void)executed;
      });
}

} // namespace flagdnn::testing
