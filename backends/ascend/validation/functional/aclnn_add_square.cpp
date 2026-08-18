/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_add_square.hpp"

#include "common/add.hpp"
#include "common/pointwise.hpp"
#include "validation/acl_runtime.hpp"
#include "validation/functional/aclnn_add.hpp"
#include "validation/functional/aclnn_binary_pointwise.hpp"
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

namespace flagdnn::testing {
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
          "ACLNN AddSquare binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument(
        "ACLNN AddSquare binding is missing or null");
  }
  return *result;
}

PointwiseTestCase make_square_case(const AddSquareTestCase& test_case,
                                   const TestTensor& right_alias,
                                   const TestTensor& square) {
  PointwiseTestCase result;
  result.name = test_case.name + "_aclnn_square";
  result.mode = FLAGDNN_POINTWISE_MUL;
  result.inputs = {test_case.right, right_alias};
  result.output = square;
  result.input_domains = {PointwiseInputDomain::kReal,
                          PointwiseInputDomain::kReal};
  result.absolute_tolerance = test_case.absolute_tolerance;
  result.relative_tolerance = test_case.relative_tolerance;
  return result;
}

AddTestCase make_add_case(const AddSquareTestCase& test_case,
                          const TestTensor& square) {
  AddTestCase result;
  result.name = test_case.name + "_aclnn_add";
  result.left = test_case.left;
  result.right = square;
  result.output = test_case.output;
  result.alpha = 1.0;
  result.absolute_tolerance = test_case.absolute_tolerance;
  result.relative_tolerance = test_case.relative_tolerance;
  return result;
}

class AclnnAddSquareExecutable final : public CompositeExecutable {
 public:
  AclnnAddSquareExecutable(AddSquareTestCase test_case,
                           AclnnAddSquarePlan plan)
      : test_case_(std::move(test_case)), plan_(std::move(plan)) {
    validate_composite_case(test_case_);
    square_case_ =
        make_square_case(test_case_, plan_.right_alias, plan_.square);
    add_case_ = make_add_case(test_case_, plan_.square);
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument("ACLNN AddSquare prepare stream is null");
    }
    if (state_ != nullptr) {
      acl::check_acl(
          aclrtSynchronizeStream(
              reinterpret_cast<aclrtStream>(state_->stream)),
          "aclrtSynchronizeStream(before ACLNN AddSquare reprepare)");
    }
    const flagdnnBinding_t& left = find_binding(bindings, plan_.left.uid);
    const flagdnnBinding_t& right = find_binding(bindings, plan_.right.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);

    auto candidate = std::make_unique<State>();
    candidate->square_buffer = std::make_unique<acl::DeviceBuffer>(
        tensor_io::allocation_byte_count(plan_.square));
    candidate->square = build_pointwise_reference(square_case_);
    candidate->add = build_add_reference(add_case_);
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
    } catch (const AclnnPointwiseUnsupportedError& error) {
      (void)aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
      throw AclnnAddSquareUnsupportedError(error.status(), error.what());
    } catch (const AclnnUnsupportedError& error) {
      (void)aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
      throw AclnnAddSquareUnsupportedError(error.status(), error.what());
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
          "ACLNN AddSquare execute called before prepare");
    }
    const flagdnnBinding_t& left = find_binding(bindings, plan_.left.uid);
    const flagdnnBinding_t& right = find_binding(bindings, plan_.right.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    if (left.device_pointer != state_->left_pointer ||
        right.device_pointer != state_->right_pointer ||
        output.device_pointer != state_->output_pointer) {
      throw std::invalid_argument(
          "ACLNN AddSquare repeatable executor binding address changed");
    }
    if (stream != state_->stream) {
      throw std::invalid_argument(
          "ACLNN AddSquare repeatable executor stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument("ACLNN AddSquare workspace is too small");
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
    // Member declaration order makes the executables die before the buffer
    // whose address their repeatable ACLNN executors retain.
    std::unique_ptr<acl::DeviceBuffer> square_buffer;
    std::unique_ptr<CompositeExecutable> square;
    std::unique_ptr<CompositeExecutable> add;
    std::vector<flagdnnBinding_t> square_bindings;
    std::vector<flagdnnBinding_t> add_bindings;
    void* left_pointer = nullptr;
    void* right_pointer = nullptr;
    void* output_pointer = nullptr;
    flagdnnStream_t stream = nullptr;
    std::size_t workspace_size = 0;
  };

  AddSquareTestCase test_case_;
  AclnnAddSquarePlan plan_;
  PointwiseTestCase square_case_;
  AddTestCase add_case_;
  std::unique_ptr<State> state_;
};

}  // namespace

AclnnAddSquarePlan plan_aclnn_add_square(
    const AddSquareTestCase& test_case) {
  validate_composite_case(test_case);
  if (test_case.output.uid >
      std::numeric_limits<std::int64_t>::max() - 2) {
    throw std::overflow_error(
        "ACLNN AddSquare generated UIDs overflow int64");
  }
  TestTensor square = test_case.output;
  square.uid = test_case.output.uid + 1;
  square.binding_byte_offset = 0;
  TestTensor right_alias = test_case.right;
  right_alias.uid = test_case.output.uid + 2;
  if (square.uid == test_case.left.uid ||
      square.uid == test_case.right.uid ||
      right_alias.uid == test_case.left.uid ||
      right_alias.uid == test_case.right.uid ||
      right_alias.uid == test_case.output.uid ||
      right_alias.uid == square.uid) {
    throw std::invalid_argument(
        "ACLNN AddSquare generated UID collides with a case tensor UID");
  }
  const AclnnBinaryPointwisePlan square_plan =
      plan_aclnn_binary_pointwise(
          make_square_case(test_case, right_alias, square));
  const AclnnAddPlan add_plan =
      plan_aclnn_add(make_add_case(test_case, square_plan.output));
  (void)tensor_io::encoded_byte_count(square_plan.output);
  return {add_plan.left,
          square_plan.left,
          square_plan.right,
          square_plan.output,
          add_plan.output};
}

std::unique_ptr<CompositeExecutable> build_add_square_reference(
    const AddSquareTestCase& test_case) {
  return std::make_unique<AclnnAddSquareExecutable>(
      test_case, plan_aclnn_add_square(test_case));
}

}  // namespace flagdnn::testing
