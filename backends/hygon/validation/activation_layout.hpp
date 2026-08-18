/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_ACTIVATION_LAYOUT_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_ACTIVATION_LAYOUT_HPP_

#include "hipdnn_reference.hpp"

#include <span>
#include <vector>

namespace flagdnn::validation::hygon {

/*
 * Returns an equivalent dense four-dimensional descriptor view when every
 * tensor participates in the same compact physical mapping.  This helper is
 * intentionally operation-agnostic; callers must restrict it to primitive
 * activation inputs/outputs after validating their original operation.
 */
[[nodiscard]] std::vector<ReferenceTensor>
hipdnn_activation_descriptor_tensors(std::span<const ReferenceTensor> tensors);

} // namespace flagdnn::validation::hygon

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_ACTIVATION_LAYOUT_HPP_
