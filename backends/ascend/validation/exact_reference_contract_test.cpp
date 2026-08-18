/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/exact_reference.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using flagdnn::validation::ascend::ExactReferenceCoverage;
using flagdnn::validation::ascend::compare_exact_reference;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template <typename Callback>
void require_failure(Callback&& callback, std::string_view needle) {
  try {
    callback();
  } catch (const std::exception& error) {
    require(std::string_view(error.what()).find(needle) !=
                std::string_view::npos,
            "exact-reference failure did not identify the contract");
    return;
  }
  throw std::runtime_error("exact-reference contract violation was accepted");
}

}  // namespace

int main() {
  try {
    ExactReferenceCoverage coverage(3);
    coverage.record_pass("case_a");
    coverage.record_skip("case_b", "ACLNN status 561002");
    coverage.record_pass("case_c");
    coverage.require_complete();
    require(coverage.required() == 3U && coverage.passed() == 2U &&
                coverage.unsupported() == 1U,
            "exact-reference coverage counters are wrong");
    require(coverage.skip_lines() ==
                std::vector<std::string>{
                    "SKIP case=case_b reason=ACLNN status 561002"},
            "exact-reference SKIP output is unstable");
    require(coverage.summary() ==
                "ACLNN_REQUIRED_SUMMARY required=3 pass=2 unsupported=1 "
                "fail=0",
            "exact-reference summary is wrong");

    require_failure(
        [&] { coverage.record_pass("case_a"); }, "duplicate");
    require_failure(
        [&] { coverage.record_skip("case_d", "unsupported"); }, "exceeds");

    ExactReferenceCoverage incomplete(2);
    incomplete.record_pass("only_case");
    require_failure([&] { incomplete.require_complete(); }, "incomplete");
    require_failure([&] { (void)incomplete.summary(); }, "incomplete");

    ExactReferenceCoverage invalid(1);
    require_failure([&] { invalid.record_pass(""); }, "case");
    require_failure(
        [&] { invalid.record_skip("case with space", "unsupported"); },
        "case");
    require_failure(
        [&] { invalid.record_skip("case", ""); }, "reason");
    require_failure(
        [&] { invalid.record_skip("case", "bad\nreason"); }, "reason");

    const std::vector<float> exact = {1.0F, -2.0F, 0.0F};
    compare_exact_reference(exact, exact, 0.0, 0.0, "exact");
    compare_exact_reference(
        std::vector<float>{1.0F},
        std::vector<float>{1.1F},
        0.11,
        0.0,
        "absolute");
    compare_exact_reference(
        std::vector<float>{100.0F},
        std::vector<float>{101.0F},
        0.5,
        0.02,
        "relative");

    require_failure(
        [&] {
          compare_exact_reference(
              std::vector<float>{1.0F},
              std::vector<float>{1.1F},
              0.01,
              0.01,
              "outside");
        },
        "mismatch");
    require_failure(
        [&] {
          compare_exact_reference(
              std::vector<float>{1.0F},
              std::vector<float>{1.0F, 2.0F},
              0.0,
              0.0,
              "size");
        },
        "size");
    for (const float nonfinite : {
             std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::infinity(),
         }) {
      require_failure(
          [&] {
            compare_exact_reference(
                std::vector<float>{nonfinite},
                std::vector<float>{0.0F},
                0.0,
                0.0,
                "nonfinite");
          },
          "finite");
    }
    require_failure(
        [&] {
          compare_exact_reference(
              std::vector<float>{0.0F},
              std::vector<float>{0.0F},
              -1.0,
              0.0,
              "tolerance");
        },
        "tolerance");

    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
