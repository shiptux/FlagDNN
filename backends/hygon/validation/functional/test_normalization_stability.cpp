/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

namespace flagdnn::testing {

int run_hygon_normalization_stability_test(int argc, char **argv);

} // namespace flagdnn::testing

int main(int argc, char **argv) {
  return flagdnn::testing::run_hygon_normalization_stability_test(argc, argv);
}
