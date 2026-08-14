/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_POINTWISE_MATCHER_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_POINTWISE_MATCHER_HPP_

#include "common/case.hpp"

#include <cmath>
#include <cstddef>

namespace flagdnn::benchmarking::hipdnn_detail {

inline bool has_default_pointwise_attributes(
    const flagdnnPointwiseAttributes_t &attributes) noexcept {
  return attributes.struct_size == sizeof(flagdnnPointwiseAttributes_t) &&
         attributes.version == FLAGDNN_POINTWISE_ATTRIBUTES_VERSION &&
         attributes.flags == 0U && attributes.relu_lower_clip == 0.0 &&
         attributes.relu_upper_clip == 0.0 &&
         attributes.relu_lower_clip_slope == 0.0 &&
         attributes.swish_beta == 1.0 && attributes.elu_alpha == 1.0 &&
         attributes.softplus_beta == 1.0;
}

inline bool same_tensor_layout(const TensorSpec &left,
                               const TensorSpec &right) noexcept {
  return left.data_type == right.data_type &&
         left.dimensions == right.dimensions && left.strides == right.strides;
}

/*
 * Recognize only the exact graph
 *
 *   square = right * right
 *   output = left + square
 *
 * that HipdnnPointwisePlan emulates. Checking only node modes is unsafe: a
 * different UID topology, alpha or active attribute would benchmark a
 * different FlagDNN graph against the add-square reference sequence.
 */
inline bool
matches_add_square_graph(const BenchmarkCase &specification) noexcept {
  if (specification.operation != Operation::kGraph ||
      specification.output_count != 1 || specification.tensors.size() != 3 ||
      specification.graph.intermediates.size() != 1 ||
      specification.graph.nodes.size() != 2) {
    return false;
  }

  const TensorSpec &left = specification.tensors[0];
  const TensorSpec &right = specification.tensors[1];
  const TensorSpec &output = specification.tensors[2];
  const TensorSpec &square = specification.graph.intermediates[0];
  if (left.uid <= 0 || right.uid <= 0 || output.uid <= 0 || square.uid <= 0 ||
      left.uid == right.uid || left.uid == output.uid ||
      left.uid == square.uid || right.uid == output.uid ||
      right.uid == square.uid || output.uid == square.uid ||
      !same_tensor_layout(square, output)) {
    return false;
  }

  const GraphNodeSpec &square_node = specification.graph.nodes[0];
  const GraphNodeSpec &add_node = specification.graph.nodes[1];
  return square_node.operation == Operation::kPointwise &&
         square_node.pointwise_mode == FLAGDNN_POINTWISE_MUL &&
         square_node.input_uids.size() == 2 &&
         square_node.input_uids[0] == right.uid &&
         square_node.input_uids[1] == right.uid &&
         square_node.output_uid == square.uid &&
         std::isfinite(square_node.alpha) && square_node.alpha == 1.0 &&
         has_default_pointwise_attributes(square_node.pointwise_attributes) &&
         add_node.operation == Operation::kPointwise &&
         add_node.pointwise_mode == FLAGDNN_POINTWISE_ADD &&
         add_node.input_uids.size() == 2 &&
         add_node.input_uids[0] == left.uid &&
         add_node.input_uids[1] == square.uid &&
         add_node.output_uid == output.uid && std::isfinite(add_node.alpha) &&
         add_node.alpha == 1.0 &&
         has_default_pointwise_attributes(add_node.pointwise_attributes);
}

} // namespace flagdnn::benchmarking::hipdnn_detail

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_POINTWISE_MATCHER_HPP_
