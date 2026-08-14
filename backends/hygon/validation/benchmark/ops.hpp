/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_OPS_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_OPS_HPP_

#include "hipdnn_provider.hpp"

#include <memory>
#include <string>
#include <vector>

namespace flagdnn::benchmarking::hipdnn_detail {

[[nodiscard]] ProviderCapability
pointwise_capability(const BenchmarkCase &specification);
[[nodiscard]] ProviderCapability
pointwise_capture_capability(const BenchmarkCase &specification);
[[nodiscard]] std::unique_ptr<HipdnnExecutable>
build_pointwise(const BenchmarkCase &specification);
[[nodiscard]] std::vector<TensorSpec>
pointwise_reference_specs(const BenchmarkCase &specification);
[[nodiscard]] std::vector<validation::hygon::ReferenceTensor>
pointwise_diagnostic_tensors(const BenchmarkCase &specification);
[[nodiscard]] std::string
pointwise_operation_name(const BenchmarkCase &specification);
[[nodiscard]] bool
pointwise_uses_sequence(const BenchmarkCase &specification) noexcept;

[[nodiscard]] ProviderCapability
tensor_capability(const BenchmarkCase &specification);
[[nodiscard]] std::unique_ptr<HipdnnExecutable>
build_tensor(const BenchmarkCase &specification);
[[nodiscard]] std::vector<TensorSpec>
tensor_reference_specs(const BenchmarkCase &specification);
[[nodiscard]] std::vector<validation::hygon::ReferenceTensor>
tensor_diagnostic_tensors(const BenchmarkCase &specification);
[[nodiscard]] std::string
tensor_operation_name(const BenchmarkCase &specification);

[[nodiscard]] bool
is_convolution_case(const BenchmarkCase &specification) noexcept;
[[nodiscard]] ProviderCapability
convolution_capability(const BenchmarkCase &specification);
[[nodiscard]] std::unique_ptr<HipdnnExecutable>
build_convolution(const BenchmarkCase &specification,
                  HipdnnReferencePolicy policy);
[[nodiscard]] bool convolution_uses_fp32_correctness_oracle(
    const BenchmarkCase &specification) noexcept;
[[nodiscard]] std::vector<TensorSpec>
convolution_reference_specs(const BenchmarkCase &specification,
                            HipdnnReferencePolicy policy);
[[nodiscard]] std::vector<validation::hygon::ReferenceTensor>
convolution_semantic_reference_tensors(const BenchmarkCase &specification,
                                       HipdnnReferencePolicy policy);
[[nodiscard]] std::vector<validation::hygon::ReferenceTensor>
convolution_diagnostic_tensors(const BenchmarkCase &specification);
[[nodiscard]] std::string
convolution_operation_name(const BenchmarkCase &specification);
[[nodiscard]] bool
convolution_uses_sequence(const BenchmarkCase &specification) noexcept;

[[nodiscard]] ProviderCapability
normalization_capability(const BenchmarkCase &specification);
[[nodiscard]] std::unique_ptr<HipdnnExecutable>
build_normalization(const BenchmarkCase &specification);
[[nodiscard]] std::vector<TensorSpec>
normalization_reference_specs(const BenchmarkCase &specification);
[[nodiscard]] std::vector<validation::hygon::ReferenceTensor>
normalization_diagnostic_tensors(const BenchmarkCase &specification);
[[nodiscard]] std::string
normalization_operation_name(const BenchmarkCase &specification);

} // namespace flagdnn::benchmarking::hipdnn_detail

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_OPS_HPP_
