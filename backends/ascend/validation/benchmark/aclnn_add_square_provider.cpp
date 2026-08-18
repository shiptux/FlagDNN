/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/benchmark/aclnn_add_square_provider.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/benchmark/aclnn_provider.hpp"
#include "validation/tensor_io.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

namespace acl = flagdnn::validation::ascend;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;

const flagdnnBinding_t& find_binding(
    std::span<const flagdnnBinding_t> bindings,
    std::int64_t uid) {
  const flagdnnBinding_t* result = nullptr;
  for (const flagdnnBinding_t& binding : bindings) {
    if (binding.uid != uid) {
      continue;
    }
    if (result != nullptr) {
      throw std::invalid_argument(
          "ACLNN AddSquare benchmark binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument(
        "ACLNN AddSquare benchmark binding is missing or null");
  }
  return *result;
}

void validate_add_square_graph(const BenchmarkCase& specification) {
  if (specification.operation != Operation::kGraph ||
      specification.tensors.size() != 3 ||
      specification.output_count != 1 ||
      specification.graph.intermediates.size() != 1 ||
      specification.graph.nodes.size() != 2) {
    throw std::invalid_argument(
        "ACLNN AddSquare benchmark requires a two-node Graph");
  }
  const TensorSpec& left = specification.tensors[0];
  const TensorSpec& right = specification.tensors[1];
  const TensorSpec& output = specification.tensors[2];
  const TensorSpec& square = specification.graph.intermediates[0];
  const GraphNodeSpec& mul = specification.graph.nodes[0];
  const GraphNodeSpec& add = specification.graph.nodes[1];
  if (mul.operation != Operation::kPointwise ||
      mul.pointwise_mode != FLAGDNN_POINTWISE_MUL ||
      mul.input_uids != std::vector<std::int64_t>({right.uid, right.uid}) ||
      mul.output_uid != square.uid || mul.alpha != 1.0 ||
      add.operation != Operation::kPointwise ||
      add.pointwise_mode != FLAGDNN_POINTWISE_ADD ||
      add.input_uids != std::vector<std::int64_t>({left.uid, square.uid}) ||
      add.output_uid != output.uid || add.alpha != 1.0) {
    throw std::invalid_argument(
        "ACLNN AddSquare benchmark Graph is not "
        "Mul(right,right)->Add(left,square)");
  }
  if (left.data_type != right.data_type ||
      left.data_type != square.data_type ||
      left.data_type != output.data_type ||
      left.dimensions != right.dimensions ||
      left.dimensions != square.dimensions ||
      left.dimensions != output.dimensions) {
    throw std::invalid_argument(
        "ACLNN AddSquare benchmark tensor metadata is inconsistent");
  }
}

BenchmarkCase make_square_case(const BenchmarkCase& specification,
                               const TensorSpec& right_alias,
                               const TensorSpec& square) {
  BenchmarkCase result;
  result.name = specification.name + "_aclnn_square";
  result.operation = Operation::kPointwise;
  result.pointwise_mode = FLAGDNN_POINTWISE_MUL;
  result.tensors = {specification.tensors[1],
                    right_alias,
                    square};
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  return result;
}

BenchmarkCase make_add_case(const BenchmarkCase& specification,
                            const TensorSpec& square) {
  BenchmarkCase result;
  result.name = specification.name + "_aclnn_add";
  result.operation = Operation::kAdd;
  result.tensors = {specification.tensors[0],
                    square,
                    specification.tensors[2]};
  result.add_alpha = 1.0;
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  return result;
}

class AclnnAddSquareExecutable final : public BenchmarkExecutable {
 public:
  AclnnAddSquareExecutable(BenchmarkCase specification,
                           AclnnAddSquareBenchmarkPlan plan)
      : specification_(std::move(specification)), plan_(std::move(plan)) {
    square_case_ =
        make_square_case(specification_, plan_.right_alias, plan_.square);
    add_case_ = make_add_case(specification_, plan_.square);
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument(
          "ACLNN AddSquare benchmark prepare stream is null");
    }
    if (state_ != nullptr) {
      acl::check_acl(
          aclrtSynchronizeStream(
              reinterpret_cast<aclrtStream>(state_->stream)),
          "aclrtSynchronizeStream(before ACLNN AddSquare benchmark reprepare)");
    }
    const flagdnnBinding_t& left = find_binding(bindings, plan_.left.uid);
    const flagdnnBinding_t& right = find_binding(bindings, plan_.right.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    auto candidate = std::make_unique<State>();
    candidate->square_buffer = std::make_unique<acl::DeviceBuffer>(
        tensor_io::allocation_byte_count(plan_.square));
    AclnnProvider provider;
    candidate->square = provider.build(square_case_);
    candidate->add = provider.build(add_case_);
    candidate->square_bindings = {
        {plan_.right.uid, right.device_pointer},
        {plan_.right_alias.uid, right.device_pointer},
        {plan_.square.uid,
         candidate->square_buffer->opaque_at(
             plan_.square.binding_byte_offset)},
    };
    candidate->add_bindings = {
        {plan_.left.uid, left.device_pointer},
        {plan_.square.uid,
         candidate->square_buffer->opaque_at(
             plan_.square.binding_byte_offset)},
        {plan_.output.uid, output.device_pointer},
    };
    try {
      candidate->square->prepare(candidate->square_bindings, stream);
      candidate->add->prepare(candidate->add_bindings, stream);
    } catch (...) {
      (void)aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
      throw;
    }
    candidate->left_pointer = left.device_pointer;
    candidate->right_pointer = right.device_pointer;
    candidate->output_pointer = output.device_pointer;
    candidate->stream = stream;
    candidate->workspace_size = std::max(candidate->square->workspace_size(),
                                         candidate->add->workspace_size());
    state_ = std::move(candidate);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return state_ == nullptr ? 0 : state_->workspace_size;
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (state_ == nullptr) {
      throw std::logic_error(
          "ACLNN AddSquare benchmark execute called before prepare");
    }
    const flagdnnBinding_t& left = find_binding(bindings, plan_.left.uid);
    const flagdnnBinding_t& right = find_binding(bindings, plan_.right.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    if (left.device_pointer != state_->left_pointer ||
        right.device_pointer != state_->right_pointer ||
        output.device_pointer != state_->output_pointer) {
      throw std::invalid_argument(
          "ACLNN AddSquare benchmark binding address changed");
    }
    if (stream != state_->stream) {
      throw std::invalid_argument(
          "ACLNN AddSquare benchmark stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "ACLNN AddSquare benchmark workspace is too small");
    }
    state_->square->execute(
        state_->square_bindings,
        state_->square->workspace_size() == 0 ? nullptr : workspace,
        state_->square->workspace_size(),
        stream);
    state_->add->execute(
        state_->add_bindings,
        state_->add->workspace_size() == 0 ? nullptr : workspace,
        state_->add->workspace_size(),
        stream);
  }

 private:
  struct State {
    // Repeatable executors retain the square address. Declaring the buffer
    // first guarantees that both executors are destroyed before it.
    std::unique_ptr<acl::DeviceBuffer> square_buffer;
    std::unique_ptr<BenchmarkExecutable> square;
    std::unique_ptr<BenchmarkExecutable> add;
    std::vector<flagdnnBinding_t> square_bindings;
    std::vector<flagdnnBinding_t> add_bindings;
    void* left_pointer = nullptr;
    void* right_pointer = nullptr;
    void* output_pointer = nullptr;
    flagdnnStream_t stream = nullptr;
    std::size_t workspace_size = 0;
  };

  BenchmarkCase specification_;
  AclnnAddSquareBenchmarkPlan plan_;
  BenchmarkCase square_case_;
  BenchmarkCase add_case_;
  std::unique_ptr<State> state_;
};

}  // namespace

AclnnAddSquareBenchmarkPlan plan_aclnn_add_square(
    const BenchmarkCase& specification) {
  validate_add_square_graph(specification);
  const TensorSpec& left = specification.tensors[0];
  const TensorSpec& right = specification.tensors[1];
  const TensorSpec& output = specification.tensors[2];
  const TensorSpec& square = specification.graph.intermediates[0];
  if (output.uid == std::numeric_limits<std::int64_t>::max()) {
    throw std::overflow_error(
        "ACLNN AddSquare benchmark alias UID overflows int64");
  }
  TensorSpec right_alias = right;
  right_alias.uid = output.uid + 1;
  if (right_alias.uid == left.uid || right_alias.uid == right.uid ||
      right_alias.uid == output.uid || right_alias.uid == square.uid) {
    throw std::invalid_argument(
        "ACLNN AddSquare benchmark alias UID collides with a Graph tensor");
  }
  const AclnnBinaryPointwiseBenchmarkPlan square_plan =
      plan_aclnn_binary_pointwise(
          make_square_case(specification, right_alias, square));
  const AclnnBinaryPointwiseBenchmarkPlan add_plan =
      plan_aclnn_binary_pointwise(
          make_add_case(specification, square_plan.output));
  return {add_plan.left,
          square_plan.left,
          square_plan.right,
          square_plan.output,
          add_plan.output};
}

ProviderCapability AclnnAddSquareProvider::capability(
    const BenchmarkCase& specification) const {
  if (specification.operation != Operation::kGraph) {
    return ProviderCapability::unsupported(
        "no exact ACLNN AddSquare composition mapping");
  }
  (void)plan_aclnn_add_square(specification);
  return {};
}

std::unique_ptr<BenchmarkExecutable> AclnnAddSquareProvider::build(
    const BenchmarkCase& specification) {
  return std::make_unique<AclnnAddSquareExecutable>(
      specification, plan_aclnn_add_square(specification));
}

}  // namespace flagdnn::benchmarking
