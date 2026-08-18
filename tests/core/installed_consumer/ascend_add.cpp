/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <acl/acl.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

namespace fe = ::flagdnn_frontend;

void check_acl(aclError status, std::string_view operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             " failed with aclError " +
                             std::to_string(status));
  }
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             status.get_message());
  }
}

class AclOwner {
 public:
  AclOwner() {
    check_acl(aclInit(nullptr), "aclInit");
    initialized_ = true;
    check_acl(aclrtSetDevice(0), "aclrtSetDevice");
    device_set_ = true;
  }

  ~AclOwner() { cleanup(false); }

  AclOwner(const AclOwner&) = delete;
  AclOwner& operator=(const AclOwner&) = delete;

  void finish() {
    cleanup(true);
    if (remaining_references_ != 0) {
      throw std::runtime_error(
          "installed Ascend consumer observed a leaked ACL reference");
    }
  }

 private:
  void cleanup(bool checked) {
    if (device_set_) {
      const aclError status = aclrtResetDevice(0);
      device_set_ = false;
      if (checked) {
        check_acl(status, "aclrtResetDevice");
      }
    }
    if (initialized_) {
      const aclError status = aclFinalizeReference(&remaining_references_);
      initialized_ = false;
      if (checked) {
        check_acl(status, "aclFinalizeReference");
      }
    }
  }

  bool initialized_ = false;
  bool device_set_ = false;
  std::uint64_t remaining_references_ = 0;
};

class Stream {
 public:
  Stream() { check_acl(aclrtCreateStream(&value_), "aclrtCreateStream"); }
  ~Stream() {
    if (value_ != nullptr) {
      (void)aclrtDestroyStream(value_);
    }
  }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  [[nodiscard]] aclrtStream get() const noexcept { return value_; }
  [[nodiscard]] flagdnnStream_t opaque() const noexcept {
    return reinterpret_cast<flagdnnStream_t>(value_);
  }

 private:
  aclrtStream value_ = nullptr;
};

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0) {
      check_acl(aclrtMalloc(&value_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc");
    }
  }

  ~DeviceBuffer() {
    if (value_ != nullptr) {
      (void)aclrtFree(value_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] void* get() const noexcept { return value_; }

  void copy_from(const void* source, aclrtStream stream) const {
    check_acl(aclrtMemcpyAsync(value_, bytes_, source, bytes_,
                               ACL_MEMCPY_HOST_TO_DEVICE, stream),
              "aclrtMemcpyAsync(H2D)");
  }

  void copy_to(void* destination, aclrtStream stream) const {
    check_acl(aclrtMemcpyAsync(destination, bytes_, value_, bytes_,
                               ACL_MEMCPY_DEVICE_TO_HOST, stream),
              "aclrtMemcpyAsync(D2H)");
  }

 private:
  void* value_ = nullptr;
  std::size_t bytes_ = 0;
};

fe::graph::Graph::Tensor make_tensor(fe::graph::Graph& graph,
                                     const char* name,
                                     std::int64_t uid) {
  return graph.tensor(fe::graph::Tensor_attributes()
                          .set_name(name)
                          .set_uid(uid)
                          .set_data_type(fe::DataType_t::FLOAT)
                          .set_dim({16, 16})
                          .set_stride({16, 1}));
}

fe::graph::Graph::Tensor make_boolean_tensor(fe::graph::Graph& graph,
                                             const char* name,
                                             std::int64_t uid) {
  return graph.tensor(fe::graph::Tensor_attributes()
                          .set_name(name)
                          .set_uid(uid)
                          .set_data_type(fe::DataType_t::BOOLEAN)
                          .set_dim({16, 16})
                          .set_stride({16, 1}));
}

void describe_intermediate(const fe::graph::Graph::Tensor& tensor,
                           const char* name,
                           std::int64_t uid) {
  tensor->set_name(name)
      .set_uid(uid)
      .set_data_type(fe::DataType_t::FLOAT)
      .set_dim({16, 16})
      .set_stride({16, 1})
      .set_is_virtual(true)
      .set_output(false);
}

void describe_boolean_intermediate(
    const fe::graph::Graph::Tensor& tensor,
    const char* name,
    std::int64_t uid) {
  tensor->set_name(name)
      .set_uid(uid)
      .set_data_type(fe::DataType_t::BOOLEAN)
      .set_dim({16, 16})
      .set_stride({16, 1})
      .set_is_virtual(true)
      .set_output(false);
}

void describe_boolean_output(const fe::graph::Graph::Tensor& tensor,
                             const char* name,
                             std::int64_t uid) {
  tensor->set_name(name)
      .set_uid(uid)
      .set_data_type(fe::DataType_t::BOOLEAN)
      .set_dim({16, 16})
      .set_stride({16, 1})
      .set_output(true);
}

void run_pointwise_graph() {
  AclOwner acl;
  {
    Stream stream;
    flagdnn::Handle handle("ascend", 0);

    fe::graph::Graph graph;
    graph.set_name("installed_ascend_pointwise_graph")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(true);
    const auto left = make_tensor(graph, "left", 1);
    const auto right = make_tensor(graph, "right", 2);
    auto sum = graph.pointwise(
        left,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("add")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(sum, "sum", 3);
    auto difference = graph.pointwise(
        sum,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("sub")
            .set_mode(fe::PointwiseMode_t::SUB)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_alpha(0.5));
    describe_intermediate(difference, "difference", 4);
    auto product = graph.pointwise(
        difference,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("mul")
            .set_mode(fe::PointwiseMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(product, "product", 5);
    auto quotient = graph.pointwise(
        product,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("div")
            .set_mode(fe::PointwiseMode_t::DIV)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(quotient, "quotient", 6);
    auto minimum = graph.pointwise(
        quotient,
        left,
        fe::graph::Pointwise_attributes()
            .set_name("min")
            .set_mode(fe::PointwiseMode_t::MIN)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(minimum, "minimum", 7);
    auto maximum = graph.pointwise(
        minimum,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("max")
            .set_mode(fe::PointwiseMode_t::MAX)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(maximum, "maximum", 8);
    auto identity = graph.pointwise(
        maximum,
        fe::graph::Pointwise_attributes()
            .set_name("identity")
            .set_mode(fe::PointwiseMode_t::IDENTITY)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(identity, "identity", 9);
    auto negated = graph.pointwise(
        identity,
        fe::graph::Pointwise_attributes()
            .set_name("neg")
            .set_mode(fe::PointwiseMode_t::NEG)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(negated, "negated", 10);
    auto exponential_linear = graph.pointwise(
        negated,
        fe::graph::Pointwise_attributes()
            .set_name("elu")
            .set_mode(fe::PointwiseMode_t::ELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_elu_alpha(0.75F));
    describe_intermediate(exponential_linear, "exponential_linear", 11);
    auto leaky = graph.pointwise(
        exponential_linear,
        fe::graph::Pointwise_attributes()
            .set_name("leaky_relu")
            .set_mode(fe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_relu_lower_clip_slope(0.2F));
    describe_intermediate(leaky, "leaky", 12);
    auto magnitude = graph.pointwise(
        leaky,
        fe::graph::Pointwise_attributes()
            .set_name("abs")
            .set_mode(fe::PointwiseMode_t::ABS)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(magnitude, "magnitude", 13);
    auto ceiled = graph.pointwise(
        magnitude,
        fe::graph::Pointwise_attributes()
            .set_name("ceil")
            .set_mode(fe::PointwiseMode_t::CEIL)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(ceiled, "ceiled", 14);
    auto floored = graph.pointwise(
        ceiled,
        fe::graph::Pointwise_attributes()
            .set_name("floor")
            .set_mode(fe::PointwiseMode_t::FLOOR)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(floored, "floored", 15);
    auto relu = graph.pointwise(
        floored,
        fe::graph::Pointwise_attributes()
            .set_name("relu")
            .set_mode(fe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(relu, "relu", 16);
    auto reciprocal = graph.pointwise(
        relu,
        fe::graph::Pointwise_attributes()
            .set_name("reciprocal")
            .set_mode(fe::PointwiseMode_t::RECIPROCAL)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(reciprocal, "reciprocal", 17);
    auto square_root = graph.pointwise(
        reciprocal,
        fe::graph::Pointwise_attributes()
            .set_name("sqrt")
            .set_mode(fe::PointwiseMode_t::SQRT)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(square_root, "square_root", 18);
    auto inverse_square_root = graph.pointwise(
        square_root,
        fe::graph::Pointwise_attributes()
            .set_name("rsqrt")
            .set_mode(fe::PointwiseMode_t::RSQRT)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(inverse_square_root, "inverse_square_root", 19);
    auto exponentiated = graph.pointwise(
        inverse_square_root,
        fe::graph::Pointwise_attributes()
            .set_name("exp")
            .set_mode(fe::PointwiseMode_t::EXP)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(exponentiated, "exponentiated", 20);
    auto logarithm = graph.pointwise(
        exponentiated,
        fe::graph::Pointwise_attributes()
            .set_name("log")
            .set_mode(fe::PointwiseMode_t::LOG)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(logarithm, "logarithm", 21);
    auto hyperbolic_tangent = graph.pointwise(
        logarithm,
        fe::graph::Pointwise_attributes()
            .set_name("tanh")
            .set_mode(fe::PointwiseMode_t::TANH_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(hyperbolic_tangent, "hyperbolic_tangent", 22);
    auto sigmoid = graph.pointwise(
        hyperbolic_tangent,
        fe::graph::Pointwise_attributes()
            .set_name("sigmoid")
            .set_mode(fe::PointwiseMode_t::SIGMOID_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(sigmoid, "sigmoid", 23);
    auto softplus = graph.pointwise(
        sigmoid,
        fe::graph::Pointwise_attributes()
            .set_name("softplus")
            .set_mode(fe::PointwiseMode_t::SOFTPLUS_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_softplus_beta(0.75F));
    describe_intermediate(softplus, "softplus", 24);
    auto swish = graph.pointwise(
        softplus,
        fe::graph::Pointwise_attributes()
            .set_name("swish")
            .set_mode(fe::PointwiseMode_t::SWISH_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_swish_beta(1.25F));
    describe_intermediate(swish, "swish", 25);
    auto gelu = graph.pointwise(
        swish,
        fe::graph::Pointwise_attributes()
            .set_name("gelu")
            .set_mode(fe::PointwiseMode_t::GELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(gelu, "gelu", 26);
    auto gelu_approx_tanh = graph.pointwise(
        gelu,
        fe::graph::Pointwise_attributes()
            .set_name("gelu_approx_tanh")
            .set_mode(fe::PointwiseMode_t::GELU_APPROX_TANH_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(gelu_approx_tanh, "gelu_approx_tanh", 27);
    auto sine = graph.pointwise(
        gelu_approx_tanh,
        fe::graph::Pointwise_attributes()
            .set_name("sin")
            .set_mode(fe::PointwiseMode_t::SIN)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(sine, "sine", 28);
    auto cosine = graph.pointwise(
        sine,
        fe::graph::Pointwise_attributes()
            .set_name("cos")
            .set_mode(fe::PointwiseMode_t::COS)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(cosine, "cosine", 29);
    auto tangent = graph.pointwise(
        cosine,
        fe::graph::Pointwise_attributes()
            .set_name("tan")
            .set_mode(fe::PointwiseMode_t::TAN)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(tangent, "tangent", 30);
    auto error_function = graph.pointwise(
        tangent,
        fe::graph::Pointwise_attributes()
            .set_name("erf")
            .set_mode(fe::PointwiseMode_t::ERF)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(error_function, "error_function", 31);
    auto modulo = graph.pointwise(
        error_function,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("mod")
            .set_mode(fe::PointwiseMode_t::MOD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(modulo, "modulo", 32);
    auto power = graph.pointwise(
        exponentiated,
        modulo,
        fe::graph::Pointwise_attributes()
            .set_name("pow")
            .set_mode(fe::PointwiseMode_t::POW)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_intermediate(power, "power", 33);
    auto output = graph.pointwise(
        power,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("sigmoid_backward")
            .set_mode(fe::PointwiseMode_t::SIGMOID_BWD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    output->set_name("output")
        .set_uid(34)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({16, 1})
        .set_output(true);
    const auto logical_a = make_boolean_tensor(graph, "logical_a", 35);
    const auto logical_b = make_boolean_tensor(graph, "logical_b", 36);
    auto logical_not_a = graph.pointwise(
        logical_a,
        fe::graph::Pointwise_attributes()
            .set_name("logical_not")
            .set_mode(fe::PointwiseMode_t::LOGICAL_NOT)
            .set_compute_data_type(fe::DataType_t::BOOLEAN));
    describe_boolean_intermediate(logical_not_a, "logical_not_a", 37);
    auto logical_and = graph.pointwise(
        logical_not_a,
        logical_b,
        fe::graph::Pointwise_attributes()
            .set_name("logical_and")
            .set_mode(fe::PointwiseMode_t::LOGICAL_AND)
            .set_compute_data_type(fe::DataType_t::BOOLEAN));
    describe_boolean_intermediate(logical_and, "logical_and", 38);
    auto logical_output = graph.pointwise(
        logical_and,
        logical_a,
        fe::graph::Pointwise_attributes()
            .set_name("logical_or")
            .set_mode(fe::PointwiseMode_t::LOGICAL_OR)
            .set_compute_data_type(fe::DataType_t::BOOLEAN));
    logical_output->set_name("logical_output")
        .set_uid(39)
        .set_data_type(fe::DataType_t::BOOLEAN)
        .set_dim({16, 16})
        .set_stride({16, 1})
        .set_output(true);
    const auto comparison_left = make_tensor(graph, "comparison_left", 40);
    const auto comparison_right = make_tensor(graph, "comparison_right", 41);
    auto comparison_equal = graph.pointwise(
        comparison_left,
        comparison_right,
        fe::graph::Pointwise_attributes()
            .set_name("cmp_eq")
            .set_mode(fe::PointwiseMode_t::CMP_EQ)
            .set_compute_data_type(fe::DataType_t::BOOLEAN));
    describe_boolean_output(comparison_equal, "comparison_equal", 42);
    auto comparison_not_equal = graph.pointwise(
        comparison_left,
        comparison_right,
        fe::graph::Pointwise_attributes()
            .set_name("cmp_neq")
            .set_mode(fe::PointwiseMode_t::CMP_NEQ)
            .set_compute_data_type(fe::DataType_t::BOOLEAN));
    describe_boolean_output(
        comparison_not_equal, "comparison_not_equal", 43);
    auto comparison_greater = graph.pointwise(
        comparison_left,
        comparison_right,
        fe::graph::Pointwise_attributes()
            .set_name("cmp_gt")
            .set_mode(fe::PointwiseMode_t::CMP_GT)
            .set_compute_data_type(fe::DataType_t::BOOLEAN));
    describe_boolean_output(comparison_greater, "comparison_greater", 44);
    auto comparison_greater_equal = graph.pointwise(
        comparison_left,
        comparison_right,
        fe::graph::Pointwise_attributes()
            .set_name("cmp_ge")
            .set_mode(fe::PointwiseMode_t::CMP_GE)
            .set_compute_data_type(fe::DataType_t::BOOLEAN));
    describe_boolean_output(
        comparison_greater_equal, "comparison_greater_equal", 45);
    auto comparison_less = graph.pointwise(
        comparison_left,
        comparison_right,
        fe::graph::Pointwise_attributes()
            .set_name("cmp_lt")
            .set_mode(fe::PointwiseMode_t::CMP_LT)
            .set_compute_data_type(fe::DataType_t::BOOLEAN));
    describe_boolean_output(comparison_less, "comparison_less", 46);
    auto comparison_less_equal = graph.pointwise(
        comparison_left,
        comparison_right,
        fe::graph::Pointwise_attributes()
            .set_name("cmp_le")
            .set_mode(fe::PointwiseMode_t::CMP_LE)
            .set_compute_data_type(fe::DataType_t::BOOLEAN));
    describe_boolean_output(
        comparison_less_equal, "comparison_less_equal", 47);
    auto binary_select = graph.pointwise(
        left,
        right,
        logical_and,
        fe::graph::Pointwise_attributes()
            .set_name("binary_select")
            .set_mode(fe::PointwiseMode_t::BINARY_SELECT)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    binary_select->set_name("binary_select")
        .set_uid(48)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({16, 1})
        .set_is_virtual(true)
        .set_output(false);
    auto logical_reshape = graph.reshape(
        binary_select,
        fe::graph::Reshape_attributes()
            .set_name("logical_reshape")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_dim({16, 16})
            .set_stride({16, 1})
            .set_reshape_mode(fe::ReshapeMode_t::LOGICAL));
    logical_reshape->set_name("logical_reshape")
        .set_uid(49)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({16, 1})
        .set_is_virtual(true)
        .set_output(false);
    auto transposed = graph.transpose(
        logical_reshape,
        fe::graph::Transpose_attributes()
            .set_name("transpose")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_permutation({1, 0}));
    transposed->set_name("transposed")
        .set_uid(50)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({1, 16})
        .set_is_virtual(true)
        .set_output(false);
    auto sliced = graph.slice(
        transposed,
        fe::graph::Slice_attributes()
            .set_name("slice")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_slices({{1, 16}, {2, 15}})
            .set_strides({3, 2}));
    sliced->set_name("layout_output")
        .set_uid(51)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({5, 7})
        .set_stride({3, 32})
        .set_output(true);
    auto reduction_sum = graph.reduction(
        sliced,
        fe::graph::Reduction_attributes()
            .set_name("reduction_sum")
            .set_mode(fe::ReductionMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_axis(1)
            .set_keep_dimensions(false));
    reduction_sum->set_name("reduction_sum")
        .set_uid(52)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({5})
        .set_stride({1})
        .set_is_virtual(true)
        .set_output(false);
    auto reduction_reshape = graph.reshape(
        reduction_sum,
        fe::graph::Reshape_attributes()
            .set_name("reduction_reshape")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_dim({5, 1})
            .set_stride({1, 1})
            .set_reshape_mode(fe::ReshapeMode_t::LOGICAL));
    reduction_reshape->set_name("reduction_reshape_output")
        .set_uid(53)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({5, 1})
        .set_stride({1, 1})
        .set_output(true);
    auto reduction_average = graph.reduction(
        reduction_reshape,
        fe::graph::Reduction_attributes()
            .set_name("reduction_avg")
            .set_mode(fe::ReductionMode_t::AVG)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_axis(0)
            .set_keep_dimensions(false));
    reduction_average->set_name("reduction_avg_output")
        .set_uid(54)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({1})
        .set_stride({1})
        .set_output(true);
    auto reduction_product = graph.reduction(
        reduction_sum,
        fe::graph::Reduction_attributes()
            .set_name("reduction_mul")
            .set_mode(fe::ReductionMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_axis(0)
            .set_keep_dimensions(false));
    reduction_product->set_name("reduction_mul_output")
        .set_uid(55)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({})
        .set_stride({})
        .set_output(true);
    auto batchnorm_input = graph.pointwise(
        left,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("batchnorm_input_add")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    batchnorm_input->set_name("batchnorm_input")
        .set_uid(56)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({16, 1})
        .set_is_virtual(true)
        .set_output(false);
    const auto make_batchnorm_parameter = [&](const char* name,
                                               std::int64_t uid) {
      return graph.tensor(fe::graph::Tensor_attributes()
                              .set_name(name)
                              .set_uid(uid)
                              .set_data_type(fe::DataType_t::FLOAT)
                              .set_dim({1, 16, 1, 1})
                              .set_stride({16, 1, 1, 1}));
    };
    const auto batchnorm_mean =
        make_batchnorm_parameter("batchnorm_mean", 57);
    const auto batchnorm_inv_variance =
        make_batchnorm_parameter("batchnorm_inv_variance", 58);
    const auto batchnorm_scale =
        make_batchnorm_parameter("batchnorm_scale", 59);
    const auto batchnorm_bias =
        make_batchnorm_parameter("batchnorm_bias", 60);
    auto batchnorm_output = graph.batchnorm_inference(
        batchnorm_input,
        batchnorm_mean,
        batchnorm_inv_variance,
        batchnorm_scale,
        batchnorm_bias,
        fe::graph::Batchnorm_inference_attributes()
            .set_name("batchnorm_inference")
            .set_compute_data_type(fe::DataType_t::FLOAT));
    batchnorm_output->set_name("batchnorm_output")
        .set_uid(61)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({20, 1})
        .set_is_virtual(true)
        .set_output(false);
    auto batchnorm_relu = graph.pointwise(
        batchnorm_output,
        fe::graph::Pointwise_attributes()
            .set_name("batchnorm_relu")
            .set_mode(fe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    batchnorm_relu->set_name("batchnorm_relu_output")
        .set_uid(62)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({24, 1})
        .set_output(true);
    auto rmsnorm_input = graph.pointwise(
        left,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("rmsnorm_input_add")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    rmsnorm_input->set_name("rmsnorm_input")
        .set_uid(63)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({16, 1})
        .set_is_virtual(true)
        .set_output(false);
    const auto rmsnorm_scale =
        graph.tensor(fe::graph::Tensor_attributes()
                         .set_name("rmsnorm_scale")
                         .set_uid(64)
                         .set_data_type(fe::DataType_t::FLOAT)
                         .set_dim({1, 16})
                         .set_stride({16, 1}));
    auto rmsnorm_bias =
        graph.tensor(fe::graph::Tensor_attributes()
                         .set_name("rmsnorm_bias")
                         .set_uid(65)
                         .set_data_type(fe::DataType_t::FLOAT)
                         .set_dim({1, 16})
                         .set_stride({16, 1}));
    auto rmsnorm_epsilon = graph.tensor(
        1.0e-3F, fe::graph::ScalarType::COMPILE_TIME_CONST);
    auto rmsnorm_outputs = graph.rmsnorm(
        rmsnorm_input,
        rmsnorm_scale,
        fe::graph::Rmsnorm_attributes()
            .set_name("rmsnorm")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
            .set_bias(rmsnorm_bias)
            .set_epsilon(rmsnorm_epsilon));
    auto rmsnorm_output = rmsnorm_outputs[0];
    rmsnorm_output->set_name("rmsnorm_output")
        .set_uid(66)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({16, 1})
        .set_is_virtual(true)
        .set_output(false);
    auto rmsnorm_inv_variance = rmsnorm_outputs[1];
    rmsnorm_inv_variance->set_name("rmsnorm_inv_variance")
        .set_uid(67)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 1})
        .set_stride({1, 1})
        .set_output(true);
    auto rmsnorm_relu = graph.pointwise(
        rmsnorm_output,
        fe::graph::Pointwise_attributes()
            .set_name("rmsnorm_relu")
            .set_mode(fe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    rmsnorm_relu->set_name("rmsnorm_relu_output")
        .set_uid(68)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({24, 1})
        .set_output(true);
    auto layernorm_input = graph.pointwise(
        left,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("layernorm_input_add")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    layernorm_input->set_name("layernorm_input")
        .set_uid(69)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({16, 1})
        .set_is_virtual(true)
        .set_output(false);
    const auto layernorm_scale =
        graph.tensor(fe::graph::Tensor_attributes()
                         .set_name("layernorm_scale")
                         .set_uid(70)
                         .set_data_type(fe::DataType_t::FLOAT)
                         .set_dim({1, 16})
                         .set_stride({16, 1}));
    const auto layernorm_bias =
        graph.tensor(fe::graph::Tensor_attributes()
                         .set_name("layernorm_bias")
                         .set_uid(71)
                         .set_data_type(fe::DataType_t::FLOAT)
                         .set_dim({1, 16})
                         .set_stride({16, 1}));
    auto layernorm_epsilon = graph.tensor(
        1.0e-3F, fe::graph::ScalarType::COMPILE_TIME_CONST);
    auto layernorm_outputs = graph.layernorm(
        layernorm_input,
        layernorm_scale,
        layernorm_bias,
        fe::graph::Layernorm_attributes()
            .set_name("layernorm")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
            .set_epsilon(layernorm_epsilon));
    auto layernorm_output = layernorm_outputs[0];
    layernorm_output->set_name("layernorm_output")
        .set_uid(72)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({16, 1})
        .set_is_virtual(true)
        .set_output(false);
    auto layernorm_mean = layernorm_outputs[1];
    layernorm_mean->set_name("layernorm_mean")
        .set_uid(73)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 1})
        .set_stride({1, 1})
        .set_output(true);
    auto layernorm_inv_variance = layernorm_outputs[2];
    layernorm_inv_variance->set_name("layernorm_inv_variance")
        .set_uid(74)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 1})
        .set_stride({1, 1})
        .set_output(true);
    auto layernorm_relu = graph.pointwise(
        layernorm_output,
        fe::graph::Pointwise_attributes()
            .set_name("layernorm_relu")
            .set_mode(fe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    layernorm_relu->set_name("layernorm_relu_output")
        .set_uid(75)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({24, 1})
        .set_output(true);
    auto batchnorm_training_input = graph.pointwise(
        left,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("batchnorm_training_input_add")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    batchnorm_training_input->set_name("batchnorm_training_input")
        .set_uid(76)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({16, 1})
        .set_is_virtual(true)
        .set_output(false);
    const auto make_batchnorm_training_parameter =
        [&](const char* name, std::int64_t uid) {
          return graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name(name)
                                  .set_uid(uid)
                                  .set_data_type(fe::DataType_t::FLOAT)
                                  .set_dim({1, 16, 1, 1})
                                  .set_stride({16, 1, 1, 1}));
        };
    const auto batchnorm_training_scale =
        make_batchnorm_training_parameter("batchnorm_training_scale", 77);
    const auto batchnorm_training_bias =
        make_batchnorm_training_parameter("batchnorm_training_bias", 78);
    auto batchnorm_training_previous_running_mean =
        make_batchnorm_training_parameter(
            "batchnorm_training_previous_running_mean", 79);
    auto batchnorm_training_previous_running_variance =
        make_batchnorm_training_parameter(
            "batchnorm_training_previous_running_variance", 80);
    auto batchnorm_training_epsilon = graph.tensor(
        1.0e-3F, fe::graph::ScalarType::COMPILE_TIME_CONST);
    auto batchnorm_training_momentum = graph.tensor(
        0.25F, fe::graph::ScalarType::COMPILE_TIME_CONST);
    auto batchnorm_training_outputs = graph.batchnorm(
        batchnorm_training_input,
        batchnorm_training_scale,
        batchnorm_training_bias,
        fe::graph::Batchnorm_attributes()
            .set_name("batchnorm_training")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_previous_running_stats(
                batchnorm_training_previous_running_mean,
                batchnorm_training_previous_running_variance,
                batchnorm_training_momentum)
            .set_epsilon(batchnorm_training_epsilon));
    auto batchnorm_training_output = batchnorm_training_outputs[0];
    batchnorm_training_output->set_name("batchnorm_training_output")
        .set_uid(81)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({16, 16})
        .set_stride({24, 1})
        .set_output(true);
    auto batchnorm_training_mean = batchnorm_training_outputs[1];
    batchnorm_training_mean->set_name("batchnorm_training_mean")
        .set_uid(82)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({1, 16, 1, 1})
        .set_stride({16, 1, 1, 1})
        .set_output(true);
    auto batchnorm_training_inv_variance = batchnorm_training_outputs[2];
    batchnorm_training_inv_variance
        ->set_name("batchnorm_training_inv_variance")
        .set_uid(83)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({1, 16, 1, 1})
        .set_stride({16, 1, 1, 1})
        .set_output(true);
    auto batchnorm_training_next_running_mean = batchnorm_training_outputs[3];
    batchnorm_training_next_running_mean
        ->set_name("batchnorm_training_next_running_mean")
        .set_uid(84)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({1, 16, 1, 1})
        .set_stride({16, 1, 1, 1})
        .set_output(true);
    auto batchnorm_training_next_running_variance =
        batchnorm_training_outputs[4];
    batchnorm_training_next_running_variance
        ->set_name("batchnorm_training_next_running_variance")
        .set_uid(85)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({1, 16, 1, 1})
        .set_stride({16, 1, 1, 1})
        .set_output(true);
    const auto matmul_a = graph.tensor(
        fe::graph::Tensor_attributes()
            .set_name("matmul_a")
            .set_uid(86)
            .set_data_type(fe::DataType_t::FLOAT)
            .set_dim({2, 3})
            .set_stride({3, 1}));
    const auto matmul_b = graph.tensor(
        fe::graph::Tensor_attributes()
            .set_name("matmul_b")
            .set_uid(87)
            .set_data_type(fe::DataType_t::FLOAT)
            .set_dim({3, 2})
            .set_stride({2, 1}));
    auto matmul_output = graph.matmul(
        matmul_a,
        matmul_b,
        fe::graph::Matmul_attributes()
            .set_name("matmul")
            .set_compute_data_type(fe::DataType_t::FLOAT));
    matmul_output->set_name("matmul_output")
        .set_uid(88)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({2, 2})
        .set_stride({2, 1})
        .set_output(true);
    const auto convolution_input = graph.tensor(
        fe::graph::Tensor_attributes()
            .set_name("convolution_input")
            .set_uid(89)
            .set_data_type(fe::DataType_t::FLOAT)
            .set_dim({1, 2, 4, 4})
            .set_stride({32, 16, 4, 1}));
    const auto convolution_filter = graph.tensor(
        fe::graph::Tensor_attributes()
            .set_name("convolution_filter")
            .set_uid(90)
            .set_data_type(fe::DataType_t::FLOAT)
            .set_dim({2, 2, 3, 3})
            .set_stride({18, 9, 3, 1}));
    auto convolution_output = graph.conv_fprop(
        convolution_input,
        convolution_filter,
        fe::graph::Conv_fprop_attributes()
            .set_name("convolution_fprop")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_padding({1, 1})
            .set_stride({1, 1})
            .set_dilation({1, 1})
            .set_convolution_mode(fe::ConvolutionMode_t::CROSS_CORRELATION)
            .set_groups(1));
    convolution_output->set_name("convolution_output")
        .set_uid(91)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({1, 2, 4, 4})
        .set_stride({32, 16, 4, 1})
        .set_output(true);
    std::cerr << "BEGIN_CREATE\n" << std::flush;
    check_frontend(graph.build(handle, {fe::HeurMode_t::A}),
                   "installed Ascend pointwise graph build");
    std::cerr << "END_CREATE\n" << std::flush;

    std::array<float, 256> host_left{};
    std::array<float, 256> host_right{};
    std::array<float, 256> host_expected{};
    std::array<float, 256> host_output{};
    std::array<std::uint8_t, 256> host_logical_a{};
    std::array<std::uint8_t, 256> host_logical_b{};
    std::array<std::uint8_t, 256> host_logical_expected{};
    std::array<std::uint8_t, 256> host_logical_output{};
    std::array<float, 256> host_comparison_left{};
    std::array<float, 256> host_comparison_right{};
    std::array<std::uint8_t, 256> host_comparison_equal_expected{};
    std::array<std::uint8_t, 256> host_comparison_not_equal_expected{};
    std::array<std::uint8_t, 256> host_comparison_greater_expected{};
    std::array<std::uint8_t, 256> host_comparison_greater_equal_expected{};
    std::array<std::uint8_t, 256> host_comparison_less_expected{};
    std::array<std::uint8_t, 256> host_comparison_less_equal_expected{};
    std::array<std::uint8_t, 256> host_comparison_equal_output{};
    std::array<std::uint8_t, 256> host_comparison_not_equal_output{};
    std::array<std::uint8_t, 256> host_comparison_greater_output{};
    std::array<std::uint8_t, 256> host_comparison_greater_equal_output{};
    std::array<std::uint8_t, 256> host_comparison_less_output{};
    std::array<std::uint8_t, 256> host_comparison_less_equal_output{};
    std::array<float, 256> host_binary_select_expected{};
    constexpr std::size_t layout_rows = 5U;
    constexpr std::size_t layout_columns = 7U;
    constexpr std::size_t layout_row_stride = 3U;
    constexpr std::size_t layout_column_stride = 32U;
    constexpr std::size_t layout_storage_elements =
        (layout_rows - 1U) * layout_row_stride +
        (layout_columns - 1U) * layout_column_stride + 1U;
    static_assert(layout_storage_elements == 205U);
    constexpr std::uint32_t layout_padding_pattern = 0xA5A5A5A5U;
    const float layout_padding =
        std::bit_cast<float>(layout_padding_pattern);
    std::array<float, layout_storage_elements> host_layout_expected{};
    std::array<float, layout_storage_elements> host_layout_output{};
    std::array<bool, layout_storage_elements> host_layout_written{};
    std::array<float, layout_rows> host_reduction_sum_expected{};
    std::array<float, layout_rows> host_reduction_reshape_output{};
    std::array<float, 1> host_reduction_average_output{};
    float host_reduction_average_expected = 0.0F;
    float host_reduction_product_expected = 1.0F;
    float host_reduction_product_output = 0.0F;
    constexpr std::size_t batchnorm_rows = 16U;
    constexpr std::size_t batchnorm_channels = 16U;
    constexpr std::size_t batchnorm_output_row_stride = 24U;
    constexpr std::size_t batchnorm_output_storage_elements =
        (batchnorm_rows - 1U) * batchnorm_output_row_stride +
        batchnorm_channels;
    static_assert(batchnorm_output_storage_elements == 376U);
    std::array<float, batchnorm_channels> host_batchnorm_mean{};
    std::array<float, batchnorm_channels> host_batchnorm_inv_variance{};
    std::array<float, batchnorm_channels> host_batchnorm_scale{};
    std::array<float, batchnorm_channels> host_batchnorm_bias{};
    std::array<float, batchnorm_output_storage_elements>
        host_batchnorm_expected{};
    std::array<float, batchnorm_output_storage_elements>
        host_batchnorm_output{};
    std::array<float, 256> host_left_after{};
    std::array<float, 256> host_right_after{};
    std::array<float, batchnorm_channels> host_batchnorm_mean_after{};
    std::array<float, batchnorm_channels>
        host_batchnorm_inv_variance_after{};
    std::array<float, batchnorm_channels> host_batchnorm_scale_after{};
    std::array<float, batchnorm_channels> host_batchnorm_bias_after{};
    constexpr std::size_t rmsnorm_rows = 16U;
    constexpr std::size_t rmsnorm_normalized_elements = 16U;
    constexpr std::size_t rmsnorm_output_row_stride = 24U;
    constexpr std::size_t rmsnorm_output_storage_elements =
        (rmsnorm_rows - 1U) * rmsnorm_output_row_stride +
        rmsnorm_normalized_elements;
    static_assert(rmsnorm_output_storage_elements == 376U);
    constexpr float rmsnorm_epsilon_value = 1.0e-3F;
    std::array<float, rmsnorm_normalized_elements> host_rmsnorm_scale{};
    std::array<float, rmsnorm_normalized_elements> host_rmsnorm_bias{};
    std::array<float, rmsnorm_rows> host_rmsnorm_inv_variance_expected{};
    std::array<float, rmsnorm_rows> host_rmsnorm_inv_variance_output{};
    std::array<float, rmsnorm_output_storage_elements>
        host_rmsnorm_expected{};
    std::array<float, rmsnorm_output_storage_elements> host_rmsnorm_output{};
    std::array<float, rmsnorm_normalized_elements>
        host_rmsnorm_scale_after{};
    std::array<float, rmsnorm_normalized_elements>
        host_rmsnorm_bias_after{};
    constexpr std::size_t layernorm_rows = 16U;
    constexpr std::size_t layernorm_normalized_elements = 16U;
    constexpr std::size_t layernorm_output_row_stride = 24U;
    constexpr std::size_t layernorm_output_storage_elements =
        (layernorm_rows - 1U) * layernorm_output_row_stride +
        layernorm_normalized_elements;
    static_assert(layernorm_output_storage_elements == 376U);
    constexpr float layernorm_epsilon_value = 1.0e-3F;
    std::array<float, layernorm_normalized_elements> host_layernorm_scale{};
    std::array<float, layernorm_normalized_elements> host_layernorm_bias{};
    std::array<float, layernorm_rows> host_layernorm_mean_expected{};
    std::array<float, layernorm_rows> host_layernorm_mean_output{};
    std::array<float, layernorm_rows>
        host_layernorm_inv_variance_expected{};
    std::array<float, layernorm_rows> host_layernorm_inv_variance_output{};
    std::array<float, layernorm_output_storage_elements>
        host_layernorm_expected{};
    std::array<float, layernorm_output_storage_elements>
        host_layernorm_output{};
    std::array<float, layernorm_normalized_elements>
        host_layernorm_scale_after{};
    std::array<float, layernorm_normalized_elements>
        host_layernorm_bias_after{};
    constexpr float batchnorm_training_epsilon_value = 1.0e-3F;
    constexpr float batchnorm_training_momentum_value = 0.25F;
    std::array<float, batchnorm_channels> host_batchnorm_training_scale{};
    std::array<float, batchnorm_channels> host_batchnorm_training_bias{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_previous_running_mean{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_previous_running_variance{};
    std::array<float, batchnorm_output_storage_elements>
        host_batchnorm_training_expected{};
    std::array<float, batchnorm_output_storage_elements>
        host_batchnorm_training_output{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_mean_expected{};
    std::array<float, batchnorm_channels> host_batchnorm_training_mean_output{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_inv_variance_expected{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_inv_variance_output{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_next_running_mean_expected{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_next_running_mean_output{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_next_running_variance_expected{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_next_running_variance_output{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_scale_after{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_bias_after{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_previous_running_mean_after{};
    std::array<float, batchnorm_channels>
        host_batchnorm_training_previous_running_variance_after{};
    constexpr std::size_t matmul_m = 2U;
    constexpr std::size_t matmul_k = 3U;
    constexpr std::size_t matmul_n = 2U;
    const std::array<float, matmul_m * matmul_k> host_matmul_a = {
        1.0F, -2.0F, 3.0F, 4.0F, 0.5F, -1.0F};
    const std::array<float, matmul_k * matmul_n> host_matmul_b = {
        2.0F, 1.0F, -1.0F, 3.0F, 0.5F, -2.0F};
    std::array<float, matmul_m * matmul_n> host_matmul_expected{};
    std::array<float, matmul_m * matmul_n> host_matmul_output{};
    std::array<float, matmul_m * matmul_k> host_matmul_a_after{};
    std::array<float, matmul_k * matmul_n> host_matmul_b_after{};
    for (std::size_t row = 0; row < matmul_m; ++row) {
      for (std::size_t column = 0; column < matmul_n; ++column) {
        double sum = 0.0;
        for (std::size_t inner = 0; inner < matmul_k; ++inner) {
          sum += static_cast<double>(host_matmul_a[row * matmul_k + inner]) *
                 static_cast<double>(
                     host_matmul_b[inner * matmul_n + column]);
        }
        host_matmul_expected[row * matmul_n + column] =
            static_cast<float>(sum);
      }
    }
    constexpr std::size_t convolution_input_channels = 2U;
    constexpr std::size_t convolution_output_channels = 2U;
    constexpr std::size_t convolution_height = 4U;
    constexpr std::size_t convolution_width = 4U;
    constexpr std::size_t convolution_kernel = 3U;
    std::array<float,
               convolution_input_channels * convolution_height *
                   convolution_width>
        host_convolution_input{};
    std::array<float,
               convolution_output_channels * convolution_input_channels *
                   convolution_kernel * convolution_kernel>
        host_convolution_filter{};
    std::array<float,
               convolution_output_channels * convolution_height *
                   convolution_width>
        host_convolution_expected{};
    std::array<float, host_convolution_expected.size()>
        host_convolution_output{};
    std::array<float, host_convolution_input.size()>
        host_convolution_input_after{};
    std::array<float, host_convolution_filter.size()>
        host_convolution_filter_after{};
    for (std::size_t index = 0; index < host_convolution_input.size();
         ++index) {
      host_convolution_input[index] =
          static_cast<float>(static_cast<int>(index % 11U) - 5) * 0.25F;
    }
    for (std::size_t index = 0; index < host_convolution_filter.size();
         ++index) {
      host_convolution_filter[index] =
          static_cast<float>(static_cast<int>(index % 7U) - 3) * 0.125F;
    }
    for (std::size_t output_channel = 0;
         output_channel < convolution_output_channels;
         ++output_channel) {
      for (std::size_t output_h = 0; output_h < convolution_height;
           ++output_h) {
        for (std::size_t output_w = 0; output_w < convolution_width;
             ++output_w) {
          double sum = 0.0;
          for (std::size_t input_channel = 0;
               input_channel < convolution_input_channels;
               ++input_channel) {
            for (std::size_t kernel_h = 0; kernel_h < convolution_kernel;
                 ++kernel_h) {
              for (std::size_t kernel_w = 0;
                   kernel_w < convolution_kernel;
                   ++kernel_w) {
                const std::int64_t input_h =
                    static_cast<std::int64_t>(output_h + kernel_h) - 1;
                const std::int64_t input_w =
                    static_cast<std::int64_t>(output_w + kernel_w) - 1;
                if (input_h < 0 || input_w < 0 ||
                    input_h >=
                        static_cast<std::int64_t>(convolution_height) ||
                    input_w >=
                        static_cast<std::int64_t>(convolution_width)) {
                  continue;
                }
                const std::size_t input_index =
                    input_channel * convolution_height * convolution_width +
                    static_cast<std::size_t>(input_h) * convolution_width +
                    static_cast<std::size_t>(input_w);
                const std::size_t filter_index =
                    ((output_channel * convolution_input_channels +
                      input_channel) *
                         convolution_kernel +
                     kernel_h) *
                        convolution_kernel +
                    kernel_w;
                sum += static_cast<double>(
                           host_convolution_input[input_index]) *
                       static_cast<double>(
                           host_convolution_filter[filter_index]);
              }
            }
          }
          const std::size_t output_index =
              output_channel * convolution_height * convolution_width +
              output_h * convolution_width + output_w;
          host_convolution_expected[output_index] = static_cast<float>(sum);
        }
      }
    }
    host_layout_expected.fill(layout_padding);
    host_layout_output.fill(layout_padding);
    host_batchnorm_expected.fill(layout_padding);
    host_batchnorm_output.fill(layout_padding);
    host_rmsnorm_expected.fill(layout_padding);
    host_rmsnorm_output.fill(layout_padding);
    host_rmsnorm_inv_variance_output.fill(layout_padding);
    host_layernorm_expected.fill(layout_padding);
    host_layernorm_output.fill(layout_padding);
    host_layernorm_mean_output.fill(layout_padding);
    host_layernorm_inv_variance_output.fill(layout_padding);
    host_batchnorm_training_expected.fill(layout_padding);
    host_batchnorm_training_output.fill(layout_padding);
    for (std::size_t index = 0; index < host_left.size(); ++index) {
      host_left[index] = static_cast<float>(index) * 0.25F;
      const int centered = static_cast<int>(index % 17U) - 8;
      host_right[index] = centered == 0
                              ? 0.5F
                              : static_cast<float>(centered) * 0.25F;
      host_logical_a[index] = static_cast<std::uint8_t>(index & 1U);
      host_logical_b[index] =
          static_cast<std::uint8_t>((index >> 1U) & 1U);
      host_logical_expected[index] = static_cast<std::uint8_t>(
          ((!static_cast<bool>(host_logical_a[index])) &&
           static_cast<bool>(host_logical_b[index])) ||
          static_cast<bool>(host_logical_a[index]));
      const float comparison_magnitude =
          static_cast<float>((index % 29U) + 1U) * 0.125F;
      switch (index % 3U) {
        case 0U:
          host_comparison_left[index] = comparison_magnitude;
          host_comparison_right[index] = comparison_magnitude;
          break;
        case 1U:
          host_comparison_left[index] = -comparison_magnitude;
          host_comparison_right[index] = comparison_magnitude;
          break;
        default:
          host_comparison_left[index] = comparison_magnitude;
          host_comparison_right[index] = -comparison_magnitude;
          break;
      }
      host_comparison_equal_expected[index] = static_cast<std::uint8_t>(
          host_comparison_left[index] == host_comparison_right[index]);
      host_comparison_not_equal_expected[index] = static_cast<std::uint8_t>(
          host_comparison_left[index] != host_comparison_right[index]);
      host_comparison_greater_expected[index] = static_cast<std::uint8_t>(
          host_comparison_left[index] > host_comparison_right[index]);
      host_comparison_greater_equal_expected[index] =
          static_cast<std::uint8_t>(
              host_comparison_left[index] >= host_comparison_right[index]);
      host_comparison_less_expected[index] = static_cast<std::uint8_t>(
          host_comparison_left[index] < host_comparison_right[index]);
      host_comparison_less_equal_expected[index] =
          static_cast<std::uint8_t>(
              host_comparison_left[index] <= host_comparison_right[index]);
    }
    constexpr std::size_t large_positive_logit_index = 255U;
    host_right[large_positive_logit_index] = 17.0F;
    for (std::size_t index = 0; index < host_expected.size(); ++index) {
      const bool select_left =
          !static_cast<bool>(host_logical_a[index]) &&
          static_cast<bool>(host_logical_b[index]);
      host_binary_select_expected[index] =
          select_left ? host_left[index] : host_right[index];
      if (host_right[index] == 0.0F) {
        throw std::runtime_error(
            "installed Ascend pointwise host oracle has a zero divisor");
      }
      const float sum_value = host_left[index] + host_right[index];
      const float difference_value = sum_value - 0.5F * host_right[index];
      const float product_value = difference_value * host_right[index];
      const float quotient_value = product_value / host_right[index];
      const float minimum_value = std::fmin(quotient_value, host_left[index]);
      const float maximum_value = std::fmax(minimum_value, host_right[index]);
      const float negated_value = -maximum_value;
      const float exponential_linear_value =
          negated_value > 0.0F
              ? negated_value
              : 0.75F * (std::exp(negated_value) - 1.0F);
      if (!std::isfinite(exponential_linear_value)) {
        throw std::runtime_error(
            "installed Ascend pointwise ELU host oracle produced a "
            "non-finite result at index " +
            std::to_string(index));
      }
      const float leaky_value =
          exponential_linear_value < 0.0F
              ? 0.2F * exponential_linear_value
              : exponential_linear_value;
      const float magnitude_value = std::abs(leaky_value);
      const float ceiled_value = std::ceil(magnitude_value);
      const float floored_value = std::floor(ceiled_value);
      const float relu_value = std::fmax(floored_value, 0.0F);
      if (!std::isfinite(relu_value) || relu_value <= 0.0F) {
        throw std::runtime_error(
            "installed Ascend pointwise host oracle left the positive "
            "reciprocal/sqrt/rsqrt domain at index " +
            std::to_string(index));
      }
      const float reciprocal_value = 1.0F / relu_value;
      if (!std::isfinite(reciprocal_value) || reciprocal_value <= 0.0F) {
        throw std::runtime_error(
            "installed Ascend pointwise host oracle produced an invalid "
            "reciprocal result at index " +
            std::to_string(index));
      }
      const float square_root_value = std::sqrt(reciprocal_value);
      if (!std::isfinite(square_root_value) || square_root_value <= 0.0F) {
        throw std::runtime_error(
            "installed Ascend pointwise host oracle produced an invalid "
            "sqrt result at index " +
            std::to_string(index));
      }
      const float inverse_square_root_value =
          1.0F / std::sqrt(square_root_value);
      if (!std::isfinite(inverse_square_root_value) ||
          inverse_square_root_value <= 0.0F) {
        throw std::runtime_error(
            "installed Ascend pointwise host oracle produced an invalid "
            "rsqrt result at index " +
            std::to_string(index));
      }
      const float exponential_value = std::exp(inverse_square_root_value);
      if (!std::isfinite(exponential_value) || exponential_value <= 0.0F) {
        throw std::runtime_error(
            "installed Ascend pointwise host oracle produced an invalid exp "
            "result for the log domain at index " +
            std::to_string(index));
      }
      const float logarithm_value = std::log(exponential_value);
      if (!std::isfinite(logarithm_value)) {
        throw std::runtime_error(
            "installed Ascend pointwise host oracle produced a non-finite "
            "log result at index " +
            std::to_string(index));
      }
      const float hyperbolic_tangent_value = std::tanh(logarithm_value);
      if (!std::isfinite(hyperbolic_tangent_value)) {
        throw std::runtime_error(
            "installed Ascend pointwise host oracle produced a non-finite "
            "tanh result at index " +
            std::to_string(index));
      }
      const float sigmoid_value =
          1.0F / (1.0F + std::exp(-hyperbolic_tangent_value));
      const float scaled_softplus = 0.75F * sigmoid_value;
      const float softplus_value =
          (std::fmax(scaled_softplus, 0.0F) +
           std::log1p(std::exp(-std::abs(scaled_softplus)))) /
          0.75F;
      const float swish_value =
          softplus_value /
          (1.0F + std::exp(-1.25F * softplus_value));
      constexpr float inverse_sqrt_two = 0.70710678118654752440F;
      const float gelu_value =
          0.5F * swish_value *
          (1.0F + std::erf(swish_value * inverse_sqrt_two));
      constexpr float gelu_tanh_scale = 0.79788456080286535588F;
      constexpr float gelu_cubic_scale = 0.044715F;
      const float gelu_approx_inner =
          gelu_tanh_scale *
          (gelu_value + gelu_cubic_scale * gelu_value * gelu_value *
                            gelu_value);
      const float gelu_approx_value =
          0.5F * gelu_value * (1.0F + std::tanh(gelu_approx_inner));
      if (!std::isfinite(sigmoid_value) || !std::isfinite(softplus_value) ||
          !std::isfinite(swish_value) || !std::isfinite(gelu_value) ||
          !std::isfinite(gelu_approx_value)) {
        throw std::runtime_error(
            "installed Ascend pointwise activation host oracle produced a "
            "non-finite result at index " +
            std::to_string(index));
      }
      const float sine_value = std::sin(gelu_approx_value);
      if (!std::isfinite(sine_value)) {
        throw std::runtime_error(
            "installed Ascend pointwise sin host oracle produced a "
            "non-finite result at index " +
            std::to_string(index));
      }
      const float cosine_value = std::cos(sine_value);
      if (!std::isfinite(cosine_value)) {
        throw std::runtime_error(
            "installed Ascend pointwise cos host oracle produced a "
            "non-finite result at index " +
            std::to_string(index));
      }
      constexpr float minimum_tan_pole_clearance = 0.25F;
      const float tan_pole_clearance = std::abs(std::cos(cosine_value));
      if (!std::isfinite(tan_pole_clearance) ||
          tan_pole_clearance < minimum_tan_pole_clearance) {
        throw std::runtime_error(
            "installed Ascend pointwise tan host oracle input is too close "
            "to a pole at index " +
            std::to_string(index));
      }
      const float tangent_value = std::tan(cosine_value);
      if (!std::isfinite(tangent_value)) {
        throw std::runtime_error(
            "installed Ascend pointwise tan host oracle produced a "
            "non-finite result at index " +
            std::to_string(index));
      }
      const float erf_value = std::erf(tangent_value);
      if (!std::isfinite(erf_value)) {
        throw std::runtime_error(
            "installed Ascend pointwise erf host oracle produced a "
            "non-finite result at index " +
            std::to_string(index));
      }
      if (!std::isfinite(host_right[index]) || host_right[index] == 0.0F) {
        throw std::runtime_error(
            "installed Ascend pointwise mod host oracle has an invalid "
            "divisor at index " +
            std::to_string(index));
      }
      const float modulo_value = std::fmod(erf_value, host_right[index]);
      if (!std::isfinite(modulo_value)) {
        throw std::runtime_error(
            "installed Ascend pointwise mod host oracle produced a "
            "non-finite result at index " +
            std::to_string(index));
      }
      if (!std::isfinite(exponential_value) || exponential_value <= 0.0F) {
        throw std::runtime_error(
            "installed Ascend pointwise pow host oracle base is not "
            "strictly positive at index " +
            std::to_string(index));
      }
      const float power_value = std::pow(exponential_value, modulo_value);
      if (!std::isfinite(power_value)) {
        throw std::runtime_error(
            "installed Ascend pointwise pow host oracle produced a "
            "non-finite result at index " +
            std::to_string(index));
      }
      const float sigmoid_backward_exponential =
          std::exp(-std::abs(host_right[index]));
      const float sigmoid_backward_inverse =
          1.0F / (1.0F + sigmoid_backward_exponential);
      const float sigmoid_backward_value =
          power_value * sigmoid_backward_exponential *
          sigmoid_backward_inverse * sigmoid_backward_inverse;
      if (!std::isfinite(sigmoid_backward_value) ||
          (index == large_positive_logit_index &&
           sigmoid_backward_value <= 0.0F)) {
        throw std::runtime_error(
            "installed Ascend pointwise sigmoid backward host oracle "
            "produced an invalid result at index " +
            std::to_string(index));
      }
      host_expected[index] = sigmoid_backward_value;
    }
    for (std::size_t channel = 0; channel < batchnorm_channels; ++channel) {
      const float centered = static_cast<float>(channel) - 7.5F;
      host_batchnorm_mean[channel] = centered * 0.5F;
      host_batchnorm_inv_variance[channel] =
          0.25F + static_cast<float>((channel % 5U) + 1U) * 0.125F;
      host_batchnorm_scale[channel] =
          channel % 2U == 0U ? 1.5F : -0.75F;
      host_batchnorm_bias[channel] =
          static_cast<float>(static_cast<int>(channel % 7U) - 3) * 0.25F;
    }
    for (std::size_t row = 0; row < batchnorm_rows; ++row) {
      for (std::size_t channel = 0; channel < batchnorm_channels; ++channel) {
        const std::size_t logical = row * batchnorm_channels + channel;
        const std::size_t physical =
            row * batchnorm_output_row_stride + channel;
        const float add_value = host_left[logical] + host_right[logical];
        const float normalized =
            (add_value - host_batchnorm_mean[channel]) *
                host_batchnorm_inv_variance[channel] *
                host_batchnorm_scale[channel] +
            host_batchnorm_bias[channel];
        host_batchnorm_expected[physical] = std::fmax(normalized, 0.0F);
      }
    }
    for (std::size_t channel = 0; channel < batchnorm_channels; ++channel) {
      host_batchnorm_training_scale[channel] =
          channel % 2U == 0U ? 1.25F : -0.625F;
      host_batchnorm_training_bias[channel] =
          static_cast<float>(static_cast<int>(channel % 7U) - 3) * 0.125F;
      host_batchnorm_training_previous_running_mean[channel] =
          static_cast<float>(static_cast<int>(channel) - 8) * 0.5F;
      host_batchnorm_training_previous_running_variance[channel] =
          0.75F + static_cast<float>(channel) * 0.25F;
      double mean = 0.0;
      for (std::size_t row = 0; row < batchnorm_rows; ++row) {
        const std::size_t logical = row * batchnorm_channels + channel;
        mean += static_cast<double>(host_left[logical] + host_right[logical]);
      }
      mean /= static_cast<double>(batchnorm_rows);
      double squared_deviations = 0.0;
      for (std::size_t row = 0; row < batchnorm_rows; ++row) {
        const std::size_t logical = row * batchnorm_channels + channel;
        const double difference =
            static_cast<double>(host_left[logical] + host_right[logical]) -
            mean;
        squared_deviations += difference * difference;
      }
      const double population_variance =
          squared_deviations / static_cast<double>(batchnorm_rows);
      const double unbiased_variance =
          squared_deviations / static_cast<double>(batchnorm_rows - 1U);
      const double inv_variance =
          1.0 / std::sqrt(population_variance +
                          batchnorm_training_epsilon_value);
      host_batchnorm_training_mean_expected[channel] =
          static_cast<float>(mean);
      host_batchnorm_training_inv_variance_expected[channel] =
          static_cast<float>(inv_variance);
      host_batchnorm_training_next_running_mean_expected[channel] =
          static_cast<float>(
              (1.0 - batchnorm_training_momentum_value) *
                  host_batchnorm_training_previous_running_mean[channel] +
              batchnorm_training_momentum_value * mean);
      host_batchnorm_training_next_running_variance_expected[channel] =
          static_cast<float>(
              (1.0 - batchnorm_training_momentum_value) *
                  host_batchnorm_training_previous_running_variance[channel] +
              batchnorm_training_momentum_value * unbiased_variance);
      for (std::size_t row = 0; row < batchnorm_rows; ++row) {
        const std::size_t logical = row * batchnorm_channels + channel;
        const std::size_t physical =
            row * batchnorm_output_row_stride + channel;
        const double input = host_left[logical] + host_right[logical];
        host_batchnorm_training_expected[physical] = static_cast<float>(
            (input - mean) * inv_variance *
                host_batchnorm_training_scale[channel] +
            host_batchnorm_training_bias[channel]);
      }
    }
    for (std::size_t column = 0;
         column < rmsnorm_normalized_elements;
         ++column) {
      host_rmsnorm_scale[column] =
          column % 2U == 0U ? 1.25F : -0.625F;
      host_rmsnorm_bias[column] =
          static_cast<float>(static_cast<int>(column % 7U) - 3) * 0.125F;
    }
    for (std::size_t row = 0; row < rmsnorm_rows; ++row) {
      float square_sum = 0.0F;
      for (std::size_t column = 0;
           column < rmsnorm_normalized_elements;
           ++column) {
        const std::size_t logical =
            row * rmsnorm_normalized_elements + column;
        const float input = host_left[logical] + host_right[logical];
        square_sum += input * input;
      }
      const float mean_square =
          square_sum / static_cast<float>(rmsnorm_normalized_elements);
      const float inv_variance =
          1.0F / std::sqrt(mean_square + rmsnorm_epsilon_value);
      if (!std::isfinite(inv_variance) || inv_variance <= 0.0F) {
        throw std::runtime_error(
            "installed Ascend RMSNorm host oracle produced an invalid "
            "inverse variance");
      }
      host_rmsnorm_inv_variance_expected[row] = inv_variance;
      for (std::size_t column = 0;
           column < rmsnorm_normalized_elements;
           ++column) {
        const std::size_t logical =
            row * rmsnorm_normalized_elements + column;
        const std::size_t physical =
            row * rmsnorm_output_row_stride + column;
        const float input = host_left[logical] + host_right[logical];
        const float normalized =
            input * inv_variance * host_rmsnorm_scale[column] +
            host_rmsnorm_bias[column];
        host_rmsnorm_expected[physical] = std::fmax(normalized, 0.0F);
      }
    }
    for (std::size_t column = 0;
         column < layernorm_normalized_elements;
         ++column) {
      host_layernorm_scale[column] =
          column % 2U == 0U ? 1.125F : -0.75F;
      host_layernorm_bias[column] =
          static_cast<float>(static_cast<int>(column % 9U) - 4) * 0.125F;
    }
    for (std::size_t row = 0; row < layernorm_rows; ++row) {
      double mean = 0.0;
      for (std::size_t column = 0;
           column < layernorm_normalized_elements;
           ++column) {
        const std::size_t logical =
            row * layernorm_normalized_elements + column;
        mean += static_cast<double>(host_left[logical] + host_right[logical]);
      }
      mean /= static_cast<double>(layernorm_normalized_elements);
      double variance = 0.0;
      for (std::size_t column = 0;
           column < layernorm_normalized_elements;
           ++column) {
        const std::size_t logical =
            row * layernorm_normalized_elements + column;
        const double centered =
            static_cast<double>(host_left[logical] + host_right[logical]) -
            mean;
        variance += centered * centered;
      }
      variance /= static_cast<double>(layernorm_normalized_elements);
      const double inv_variance =
          1.0 / std::sqrt(variance + layernorm_epsilon_value);
      if (!std::isfinite(mean) || !std::isfinite(inv_variance) ||
          inv_variance <= 0.0) {
        throw std::runtime_error(
            "installed Ascend LayerNorm host oracle produced invalid "
            "statistics");
      }
      host_layernorm_mean_expected[row] = static_cast<float>(mean);
      host_layernorm_inv_variance_expected[row] =
          static_cast<float>(inv_variance);
      for (std::size_t column = 0;
           column < layernorm_normalized_elements;
           ++column) {
        const std::size_t logical =
            row * layernorm_normalized_elements + column;
        const std::size_t physical =
            row * layernorm_output_row_stride + column;
        const double input = host_left[logical] + host_right[logical];
        const double normalized =
            (input - mean) * inv_variance * host_layernorm_scale[column] +
            host_layernorm_bias[column];
        host_layernorm_expected[physical] =
            std::fmax(static_cast<float>(normalized), 0.0F);
      }
    }
    constexpr std::size_t reshape_extent = 16U;
    constexpr std::size_t slice_row_start = 1U;
    constexpr std::size_t slice_column_start = 2U;
    constexpr std::size_t slice_row_step = 3U;
    constexpr std::size_t slice_column_step = 2U;
    for (std::size_t row = 0; row < layout_rows; ++row) {
      const std::size_t transposed_row =
          slice_row_start + row * slice_row_step;
      for (std::size_t column = 0; column < layout_columns; ++column) {
        const std::size_t transposed_column =
            slice_column_start + column * slice_column_step;
        // reshape[r, c] is binary_select[r * 16 + c], and transpose
        // maps output[i, j] to reshape[j, i].
        const std::size_t binary_select_index =
            transposed_column * reshape_extent + transposed_row;
        const std::size_t physical_output_index =
            row * layout_row_stride + column * layout_column_stride;
        if (binary_select_index >= host_binary_select_expected.size() ||
            physical_output_index >= host_layout_expected.size() ||
            host_layout_written[physical_output_index]) {
          throw std::runtime_error(
              "installed Ascend layout host oracle mapping is invalid");
        }
        const float logical_value =
            host_binary_select_expected[binary_select_index];
        host_layout_expected[physical_output_index] = logical_value;
        host_layout_written[physical_output_index] = true;
        host_reduction_sum_expected[row] += logical_value;
      }
    }
    for (const float row_sum : host_reduction_sum_expected) {
      if (!std::isfinite(row_sum) || row_sum == 0.0F) {
        throw std::runtime_error(
            "installed Ascend reduction host oracle produced an invalid "
            "row sum");
      }
      host_reduction_average_expected += row_sum;
      host_reduction_product_expected *= row_sum;
    }
    host_reduction_average_expected /= static_cast<float>(layout_rows);
    if (!std::isfinite(host_reduction_average_expected) ||
        !std::isfinite(host_reduction_product_expected)) {
      throw std::runtime_error(
          "installed Ascend reduction host oracle produced a non-finite "
          "result");
    }
    constexpr std::size_t float_tensor_bytes = sizeof(host_left);
    constexpr std::size_t logical_tensor_bytes = sizeof(host_logical_a);
    constexpr std::size_t comparison_tensor_bytes =
        sizeof(host_comparison_equal_output);
    DeviceBuffer device_left(float_tensor_bytes);
    DeviceBuffer device_right(float_tensor_bytes);
    DeviceBuffer device_output(float_tensor_bytes);
    DeviceBuffer device_logical_a(logical_tensor_bytes);
    DeviceBuffer device_logical_b(logical_tensor_bytes);
    DeviceBuffer device_logical_output(logical_tensor_bytes);
    DeviceBuffer device_comparison_left(float_tensor_bytes);
    DeviceBuffer device_comparison_right(float_tensor_bytes);
    DeviceBuffer device_comparison_equal(comparison_tensor_bytes);
    DeviceBuffer device_comparison_not_equal(comparison_tensor_bytes);
    DeviceBuffer device_comparison_greater(comparison_tensor_bytes);
    DeviceBuffer device_comparison_greater_equal(comparison_tensor_bytes);
    DeviceBuffer device_comparison_less(comparison_tensor_bytes);
    DeviceBuffer device_comparison_less_equal(comparison_tensor_bytes);
    DeviceBuffer device_layout_output(sizeof(host_layout_output));
    DeviceBuffer device_reduction_reshape(
        sizeof(host_reduction_reshape_output));
    DeviceBuffer device_reduction_average(
        sizeof(host_reduction_average_output));
    DeviceBuffer device_reduction_product(
        sizeof(host_reduction_product_output));
    DeviceBuffer device_batchnorm_mean(sizeof(host_batchnorm_mean));
    DeviceBuffer device_batchnorm_inv_variance(
        sizeof(host_batchnorm_inv_variance));
    DeviceBuffer device_batchnorm_scale(sizeof(host_batchnorm_scale));
    DeviceBuffer device_batchnorm_bias(sizeof(host_batchnorm_bias));
    DeviceBuffer device_batchnorm_output(sizeof(host_batchnorm_output));
    DeviceBuffer device_rmsnorm_scale(sizeof(host_rmsnorm_scale));
    DeviceBuffer device_rmsnorm_bias(sizeof(host_rmsnorm_bias));
    DeviceBuffer device_rmsnorm_inv_variance(
        sizeof(host_rmsnorm_inv_variance_output));
    DeviceBuffer device_rmsnorm_output(sizeof(host_rmsnorm_output));
    DeviceBuffer device_layernorm_scale(sizeof(host_layernorm_scale));
    DeviceBuffer device_layernorm_bias(sizeof(host_layernorm_bias));
    DeviceBuffer device_layernorm_mean(sizeof(host_layernorm_mean_output));
    DeviceBuffer device_layernorm_inv_variance(
        sizeof(host_layernorm_inv_variance_output));
    DeviceBuffer device_layernorm_output(sizeof(host_layernorm_output));
    DeviceBuffer device_batchnorm_training_scale(
        sizeof(host_batchnorm_training_scale));
    DeviceBuffer device_batchnorm_training_bias(
        sizeof(host_batchnorm_training_bias));
    DeviceBuffer device_batchnorm_training_previous_running_mean(
        sizeof(host_batchnorm_training_previous_running_mean));
    DeviceBuffer device_batchnorm_training_previous_running_variance(
        sizeof(host_batchnorm_training_previous_running_variance));
    DeviceBuffer device_batchnorm_training_output(
        sizeof(host_batchnorm_training_output));
    DeviceBuffer device_batchnorm_training_mean(
        sizeof(host_batchnorm_training_mean_output));
    DeviceBuffer device_batchnorm_training_inv_variance(
        sizeof(host_batchnorm_training_inv_variance_output));
    DeviceBuffer device_batchnorm_training_next_running_mean(
        sizeof(host_batchnorm_training_next_running_mean_output));
    DeviceBuffer device_batchnorm_training_next_running_variance(
        sizeof(host_batchnorm_training_next_running_variance_output));
    DeviceBuffer device_matmul_a(sizeof(host_matmul_a));
    DeviceBuffer device_matmul_b(sizeof(host_matmul_b));
    DeviceBuffer device_matmul_output(sizeof(host_matmul_output));
    DeviceBuffer device_convolution_input(sizeof(host_convolution_input));
    DeviceBuffer device_convolution_filter(sizeof(host_convolution_filter));
    DeviceBuffer device_convolution_output(sizeof(host_convolution_output));
    const std::int64_t workspace_size = graph.get_workspace_size();
    if (workspace_size <= 0) {
      throw std::runtime_error(
          "installed pointwise graph did not allocate virtual workspace");
    }
    DeviceBuffer workspace(static_cast<std::size_t>(workspace_size));
    if (workspace_size != 0 &&
        reinterpret_cast<std::uintptr_t>(workspace.get()) % 256U != 0U) {
      throw std::runtime_error(
          "installed pointwise graph workspace is not 256-byte aligned");
    }

    device_left.copy_from(host_left.data(), stream.get());
    device_right.copy_from(host_right.data(), stream.get());
    device_logical_a.copy_from(host_logical_a.data(), stream.get());
    device_logical_b.copy_from(host_logical_b.data(), stream.get());
    device_comparison_left.copy_from(
        host_comparison_left.data(), stream.get());
    device_comparison_right.copy_from(
        host_comparison_right.data(), stream.get());
    device_layout_output.copy_from(host_layout_output.data(), stream.get());
    device_batchnorm_mean.copy_from(host_batchnorm_mean.data(), stream.get());
    device_batchnorm_inv_variance.copy_from(
        host_batchnorm_inv_variance.data(), stream.get());
    device_batchnorm_scale.copy_from(host_batchnorm_scale.data(), stream.get());
    device_batchnorm_bias.copy_from(host_batchnorm_bias.data(), stream.get());
    device_batchnorm_output.copy_from(
        host_batchnorm_output.data(), stream.get());
    device_rmsnorm_scale.copy_from(
        host_rmsnorm_scale.data(), stream.get());
    device_rmsnorm_bias.copy_from(host_rmsnorm_bias.data(), stream.get());
    device_rmsnorm_inv_variance.copy_from(
        host_rmsnorm_inv_variance_output.data(), stream.get());
    device_rmsnorm_output.copy_from(
        host_rmsnorm_output.data(), stream.get());
    device_layernorm_scale.copy_from(
        host_layernorm_scale.data(), stream.get());
    device_layernorm_bias.copy_from(
        host_layernorm_bias.data(), stream.get());
    device_layernorm_mean.copy_from(
        host_layernorm_mean_output.data(), stream.get());
    device_layernorm_inv_variance.copy_from(
        host_layernorm_inv_variance_output.data(), stream.get());
    device_layernorm_output.copy_from(
        host_layernorm_output.data(), stream.get());
    device_batchnorm_training_scale.copy_from(
        host_batchnorm_training_scale.data(), stream.get());
    device_batchnorm_training_bias.copy_from(
        host_batchnorm_training_bias.data(), stream.get());
    device_batchnorm_training_previous_running_mean.copy_from(
        host_batchnorm_training_previous_running_mean.data(), stream.get());
    device_batchnorm_training_previous_running_variance.copy_from(
        host_batchnorm_training_previous_running_variance.data(), stream.get());
    device_batchnorm_training_output.copy_from(
        host_batchnorm_training_output.data(), stream.get());
    device_batchnorm_training_mean.copy_from(
        host_batchnorm_training_mean_output.data(), stream.get());
    device_batchnorm_training_inv_variance.copy_from(
        host_batchnorm_training_inv_variance_output.data(), stream.get());
    device_batchnorm_training_next_running_mean.copy_from(
        host_batchnorm_training_next_running_mean_output.data(), stream.get());
    device_batchnorm_training_next_running_variance.copy_from(
        host_batchnorm_training_next_running_variance_output.data(),
        stream.get());
    device_matmul_a.copy_from(host_matmul_a.data(), stream.get());
    device_matmul_b.copy_from(host_matmul_b.data(), stream.get());
    device_matmul_output.copy_from(host_matmul_output.data(), stream.get());
    device_convolution_input.copy_from(
        host_convolution_input.data(), stream.get());
    device_convolution_filter.copy_from(
        host_convolution_filter.data(), stream.get());
    device_convolution_output.copy_from(
        host_convolution_output.data(), stream.get());
    const std::array<flagdnnBinding_t, 47> bindings = {
        flagdnnBinding_t{1, device_left.get()},
        flagdnnBinding_t{2, device_right.get()},
        flagdnnBinding_t{34, device_output.get()},
        flagdnnBinding_t{35, device_logical_a.get()},
        flagdnnBinding_t{36, device_logical_b.get()},
        flagdnnBinding_t{39, device_logical_output.get()},
        flagdnnBinding_t{40, device_comparison_left.get()},
        flagdnnBinding_t{41, device_comparison_right.get()},
        flagdnnBinding_t{42, device_comparison_equal.get()},
        flagdnnBinding_t{43, device_comparison_not_equal.get()},
        flagdnnBinding_t{44, device_comparison_greater.get()},
        flagdnnBinding_t{45, device_comparison_greater_equal.get()},
        flagdnnBinding_t{46, device_comparison_less.get()},
        flagdnnBinding_t{47, device_comparison_less_equal.get()},
        flagdnnBinding_t{51, device_layout_output.get()},
        flagdnnBinding_t{53, device_reduction_reshape.get()},
        flagdnnBinding_t{54, device_reduction_average.get()},
        flagdnnBinding_t{55, device_reduction_product.get()},
        flagdnnBinding_t{57, device_batchnorm_mean.get()},
        flagdnnBinding_t{58, device_batchnorm_inv_variance.get()},
        flagdnnBinding_t{59, device_batchnorm_scale.get()},
        flagdnnBinding_t{60, device_batchnorm_bias.get()},
        flagdnnBinding_t{62, device_batchnorm_output.get()},
        flagdnnBinding_t{64, device_rmsnorm_scale.get()},
        flagdnnBinding_t{65, device_rmsnorm_bias.get()},
        flagdnnBinding_t{67, device_rmsnorm_inv_variance.get()},
        flagdnnBinding_t{68, device_rmsnorm_output.get()},
        flagdnnBinding_t{70, device_layernorm_scale.get()},
        flagdnnBinding_t{71, device_layernorm_bias.get()},
        flagdnnBinding_t{73, device_layernorm_mean.get()},
        flagdnnBinding_t{74, device_layernorm_inv_variance.get()},
        flagdnnBinding_t{75, device_layernorm_output.get()},
        flagdnnBinding_t{77, device_batchnorm_training_scale.get()},
        flagdnnBinding_t{78, device_batchnorm_training_bias.get()},
        flagdnnBinding_t{79,
                         device_batchnorm_training_previous_running_mean.get()},
        flagdnnBinding_t{
            80, device_batchnorm_training_previous_running_variance.get()},
        flagdnnBinding_t{81, device_batchnorm_training_output.get()},
        flagdnnBinding_t{82, device_batchnorm_training_mean.get()},
        flagdnnBinding_t{83, device_batchnorm_training_inv_variance.get()},
        flagdnnBinding_t{84,
                         device_batchnorm_training_next_running_mean.get()},
        flagdnnBinding_t{
            85, device_batchnorm_training_next_running_variance.get()},
        flagdnnBinding_t{86, device_matmul_a.get()},
        flagdnnBinding_t{87, device_matmul_b.get()},
        flagdnnBinding_t{88, device_matmul_output.get()},
        flagdnnBinding_t{89, device_convolution_input.get()},
        flagdnnBinding_t{90, device_convolution_filter.get()},
        flagdnnBinding_t{91, device_convolution_output.get()}};
    std::cerr << "BEGIN_EXECUTE\n" << std::flush;
    check_frontend(graph.execute(handle,
                                 bindings,
                                 workspace.get(),
                                 static_cast<std::size_t>(workspace_size),
                                 stream.opaque()),
                   "installed Ascend pointwise graph execute");
    std::cerr << "END_EXECUTE\n" << std::flush;
    device_output.copy_to(host_output.data(), stream.get());
    device_logical_output.copy_to(host_logical_output.data(), stream.get());
    device_comparison_equal.copy_to(
        host_comparison_equal_output.data(), stream.get());
    device_comparison_not_equal.copy_to(
        host_comparison_not_equal_output.data(), stream.get());
    device_comparison_greater.copy_to(
        host_comparison_greater_output.data(), stream.get());
    device_comparison_greater_equal.copy_to(
        host_comparison_greater_equal_output.data(), stream.get());
    device_comparison_less.copy_to(
        host_comparison_less_output.data(), stream.get());
    device_comparison_less_equal.copy_to(
        host_comparison_less_equal_output.data(), stream.get());
    device_layout_output.copy_to(host_layout_output.data(), stream.get());
    device_reduction_reshape.copy_to(
        host_reduction_reshape_output.data(), stream.get());
    device_reduction_average.copy_to(
        host_reduction_average_output.data(), stream.get());
    device_reduction_product.copy_to(
        &host_reduction_product_output, stream.get());
    device_batchnorm_output.copy_to(
        host_batchnorm_output.data(), stream.get());
    device_left.copy_to(host_left_after.data(), stream.get());
    device_right.copy_to(host_right_after.data(), stream.get());
    device_batchnorm_mean.copy_to(
        host_batchnorm_mean_after.data(), stream.get());
    device_batchnorm_inv_variance.copy_to(
        host_batchnorm_inv_variance_after.data(), stream.get());
    device_batchnorm_scale.copy_to(
        host_batchnorm_scale_after.data(), stream.get());
    device_batchnorm_bias.copy_to(
        host_batchnorm_bias_after.data(), stream.get());
    device_rmsnorm_output.copy_to(
        host_rmsnorm_output.data(), stream.get());
    device_rmsnorm_inv_variance.copy_to(
        host_rmsnorm_inv_variance_output.data(), stream.get());
    device_rmsnorm_scale.copy_to(
        host_rmsnorm_scale_after.data(), stream.get());
    device_rmsnorm_bias.copy_to(
        host_rmsnorm_bias_after.data(), stream.get());
    device_layernorm_output.copy_to(
        host_layernorm_output.data(), stream.get());
    device_layernorm_mean.copy_to(
        host_layernorm_mean_output.data(), stream.get());
    device_layernorm_inv_variance.copy_to(
        host_layernorm_inv_variance_output.data(), stream.get());
    device_layernorm_scale.copy_to(
        host_layernorm_scale_after.data(), stream.get());
    device_layernorm_bias.copy_to(
        host_layernorm_bias_after.data(), stream.get());
    device_batchnorm_training_output.copy_to(
        host_batchnorm_training_output.data(), stream.get());
    device_batchnorm_training_mean.copy_to(
        host_batchnorm_training_mean_output.data(), stream.get());
    device_batchnorm_training_inv_variance.copy_to(
        host_batchnorm_training_inv_variance_output.data(), stream.get());
    device_batchnorm_training_next_running_mean.copy_to(
        host_batchnorm_training_next_running_mean_output.data(), stream.get());
    device_batchnorm_training_next_running_variance.copy_to(
        host_batchnorm_training_next_running_variance_output.data(),
        stream.get());
    device_batchnorm_training_scale.copy_to(
        host_batchnorm_training_scale_after.data(), stream.get());
    device_batchnorm_training_bias.copy_to(
        host_batchnorm_training_bias_after.data(), stream.get());
    device_batchnorm_training_previous_running_mean.copy_to(
        host_batchnorm_training_previous_running_mean_after.data(),
        stream.get());
    device_batchnorm_training_previous_running_variance.copy_to(
        host_batchnorm_training_previous_running_variance_after.data(),
        stream.get());
    device_matmul_output.copy_to(host_matmul_output.data(), stream.get());
    device_matmul_a.copy_to(host_matmul_a_after.data(), stream.get());
    device_matmul_b.copy_to(host_matmul_b_after.data(), stream.get());
    device_convolution_output.copy_to(
        host_convolution_output.data(), stream.get());
    device_convolution_input.copy_to(
        host_convolution_input_after.data(), stream.get());
    device_convolution_filter.copy_to(
        host_convolution_filter_after.data(), stream.get());
    check_acl(aclrtSynchronizeStream(stream.get()),
              "aclrtSynchronizeStream");

    constexpr float absolute_tolerance = 2.0e-2F;
    constexpr float relative_tolerance = 1.0e-2F;
    for (std::size_t index = 0; index < host_output.size(); ++index) {
      const float expected = host_expected[index];
      const float absolute_error = std::abs(host_output[index] - expected);
      const float tolerance =
          absolute_tolerance + relative_tolerance * std::abs(expected);
      if (!std::isfinite(host_output[index]) || !std::isfinite(expected) ||
          !std::isfinite(absolute_error) || absolute_error > tolerance ||
          (index == large_positive_logit_index &&
           host_output[index] <= 0.0F)) {
        throw std::runtime_error(
            "installed Ascend pointwise graph output differs at index " +
            std::to_string(index) + ": actual=" +
            std::to_string(host_output[index]) + ", expected=" +
            std::to_string(expected) + ", absolute error=" +
            std::to_string(absolute_error) + ", tolerance=" +
            std::to_string(tolerance));
      }
    }
    for (std::size_t index = 0; index < host_logical_output.size(); ++index) {
      const std::uint8_t actual = host_logical_output[index];
      const std::uint8_t expected = host_logical_expected[index];
      if (actual > 1U || actual != expected) {
        throw std::runtime_error(
            "installed Ascend logical graph output differs at index " +
            std::to_string(index) + ": actual=" +
            std::to_string(static_cast<unsigned int>(actual)) +
            ", expected=" +
            std::to_string(static_cast<unsigned int>(expected)));
      }
    }
    for (std::size_t index = 0; index < host_layout_output.size(); ++index) {
      const std::uint32_t actual =
          std::bit_cast<std::uint32_t>(host_layout_output[index]);
      const std::uint32_t expected =
          std::bit_cast<std::uint32_t>(host_layout_expected[index]);
      if (actual != expected) {
        throw std::runtime_error(
            std::string("installed Ascend layout ") +
            (host_layout_written[index] ? "logical output" : "padding") +
            " differs at physical index " + std::to_string(index) +
            ": actual bits=" + std::to_string(actual) +
            ", expected bits=" + std::to_string(expected));
      }
    }
    constexpr float reduction_absolute_tolerance = 1.0e-5F;
    constexpr float reduction_relative_tolerance = 1.0e-5F;
    const auto require_reduction_close = [&](std::string_view name,
                                             std::size_t index,
                                             float actual,
                                             float expected) {
      const float absolute_error = std::abs(actual - expected);
      const float tolerance = reduction_absolute_tolerance +
                              reduction_relative_tolerance *
                                  std::abs(expected);
      if (!std::isfinite(actual) || !std::isfinite(expected) ||
          !std::isfinite(absolute_error) || absolute_error > tolerance) {
        throw std::runtime_error(
            "installed Ascend " + std::string(name) +
            " output differs at index " + std::to_string(index) +
            ": actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected) +
            ", absolute error=" + std::to_string(absolute_error) +
            ", tolerance=" + std::to_string(tolerance));
      }
    };
    for (std::size_t index = 0;
         index < host_reduction_reshape_output.size();
         ++index) {
      require_reduction_close("ReductionSum/Reshape",
                              index,
                              host_reduction_reshape_output[index],
                              host_reduction_sum_expected[index]);
    }
    require_reduction_close("ReductionAvg",
                            0,
                            host_reduction_average_output[0],
                            host_reduction_average_expected);
    require_reduction_close("ReductionMul rank-0",
                            0,
                            host_reduction_product_output,
                            host_reduction_product_expected);
    std::array<bool, batchnorm_output_storage_elements>
        batchnorm_output_written{};
    for (std::size_t row = 0; row < batchnorm_rows; ++row) {
      for (std::size_t channel = 0; channel < batchnorm_channels; ++channel) {
        const std::size_t physical =
            row * batchnorm_output_row_stride + channel;
        batchnorm_output_written[physical] = true;
        const float expected = host_batchnorm_expected[physical];
        const float actual = host_batchnorm_output[physical];
        const float absolute_error = std::abs(actual - expected);
        const float tolerance = 1.0e-5F + 1.0e-5F * std::abs(expected);
        if (!std::isfinite(actual) || !std::isfinite(expected) ||
            !std::isfinite(absolute_error) || absolute_error > tolerance) {
          throw std::runtime_error(
              "installed Ascend BatchNormInference/ReLU output differs at "
              "logical index " +
              std::to_string(row * batchnorm_channels + channel));
        }
      }
    }
    for (std::size_t physical = 0;
         physical < host_batchnorm_output.size();
         ++physical) {
      if (!batchnorm_output_written[physical] &&
          std::bit_cast<std::uint32_t>(host_batchnorm_output[physical]) !=
              layout_padding_pattern) {
        throw std::runtime_error(
            "installed Ascend BatchNormInference/ReLU modified output "
            "padding at physical index " +
            std::to_string(physical));
      }
    }
    if (host_left_after != host_left || host_right_after != host_right ||
        host_batchnorm_mean_after != host_batchnorm_mean ||
        host_batchnorm_inv_variance_after != host_batchnorm_inv_variance ||
        host_batchnorm_scale_after != host_batchnorm_scale ||
        host_batchnorm_bias_after != host_batchnorm_bias) {
      throw std::runtime_error(
          "installed Ascend BatchNormInference graph modified an input");
    }
    std::array<bool, rmsnorm_output_storage_elements>
        rmsnorm_output_written{};
    for (std::size_t row = 0; row < rmsnorm_rows; ++row) {
      const float expected_inv = host_rmsnorm_inv_variance_expected[row];
      const float actual_inv = host_rmsnorm_inv_variance_output[row];
      const float inv_error = std::abs(actual_inv - expected_inv);
      const float inv_tolerance =
          2.0e-4F + 2.0e-4F * std::abs(expected_inv);
      if (!std::isfinite(actual_inv) || !std::isfinite(inv_error) ||
          inv_error > inv_tolerance) {
        throw std::runtime_error(
            "installed Ascend RMSNorm inverse variance differs at row " +
            std::to_string(row));
      }
      for (std::size_t column = 0;
           column < rmsnorm_normalized_elements;
           ++column) {
        const std::size_t physical =
            row * rmsnorm_output_row_stride + column;
        rmsnorm_output_written[physical] = true;
        const float expected = host_rmsnorm_expected[physical];
        const float actual = host_rmsnorm_output[physical];
        const float absolute_error = std::abs(actual - expected);
        const float tolerance =
            2.0e-4F + 2.0e-4F * std::abs(expected);
        if (!std::isfinite(actual) || !std::isfinite(expected) ||
            !std::isfinite(absolute_error) || absolute_error > tolerance) {
          throw std::runtime_error(
              "installed Ascend RMSNorm/ReLU output differs at logical "
              "index " +
              std::to_string(row * rmsnorm_normalized_elements + column));
        }
      }
    }
    for (std::size_t physical = 0;
         physical < host_rmsnorm_output.size();
         ++physical) {
      if (!rmsnorm_output_written[physical] &&
          std::bit_cast<std::uint32_t>(host_rmsnorm_output[physical]) !=
              layout_padding_pattern) {
        throw std::runtime_error(
            "installed Ascend RMSNorm/ReLU modified output padding at "
            "physical index " +
            std::to_string(physical));
      }
    }
    if (host_rmsnorm_scale_after != host_rmsnorm_scale ||
        host_rmsnorm_bias_after != host_rmsnorm_bias) {
      throw std::runtime_error(
          "installed Ascend RMSNorm graph modified an input");
    }
    std::array<bool, layernorm_output_storage_elements>
        layernorm_output_written{};
    for (std::size_t row = 0; row < layernorm_rows; ++row) {
      const float expected_mean = host_layernorm_mean_expected[row];
      const float actual_mean = host_layernorm_mean_output[row];
      const float mean_error = std::abs(actual_mean - expected_mean);
      const float expected_inv = host_layernorm_inv_variance_expected[row];
      const float actual_inv = host_layernorm_inv_variance_output[row];
      const float inv_error = std::abs(actual_inv - expected_inv);
      if (!std::isfinite(actual_mean) || !std::isfinite(mean_error) ||
          mean_error > 2.0e-4F + 2.0e-4F * std::abs(expected_mean)) {
        throw std::runtime_error(
            "installed Ascend LayerNorm mean differs at row " +
            std::to_string(row));
      }
      if (!std::isfinite(actual_inv) || !std::isfinite(inv_error) ||
          inv_error > 2.0e-4F + 2.0e-4F * std::abs(expected_inv)) {
        throw std::runtime_error(
            "installed Ascend LayerNorm inverse variance differs at row " +
            std::to_string(row));
      }
      for (std::size_t column = 0;
           column < layernorm_normalized_elements;
           ++column) {
        const std::size_t physical =
            row * layernorm_output_row_stride + column;
        layernorm_output_written[physical] = true;
        const float expected = host_layernorm_expected[physical];
        const float actual = host_layernorm_output[physical];
        const float absolute_error = std::abs(actual - expected);
        const float tolerance =
            2.0e-4F + 2.0e-4F * std::abs(expected);
        if (!std::isfinite(actual) || !std::isfinite(expected) ||
            !std::isfinite(absolute_error) || absolute_error > tolerance) {
          throw std::runtime_error(
              "installed Ascend LayerNorm/ReLU output differs at logical "
              "index " +
              std::to_string(row * layernorm_normalized_elements + column));
        }
      }
    }
    for (std::size_t physical = 0;
         physical < host_layernorm_output.size();
         ++physical) {
      if (!layernorm_output_written[physical] &&
          std::bit_cast<std::uint32_t>(host_layernorm_output[physical]) !=
              layout_padding_pattern) {
        throw std::runtime_error(
            "installed Ascend LayerNorm/ReLU modified output padding at "
            "physical index " +
            std::to_string(physical));
      }
    }
    if (host_layernorm_scale_after != host_layernorm_scale ||
        host_layernorm_bias_after != host_layernorm_bias) {
      throw std::runtime_error(
          "installed Ascend LayerNorm graph modified an input");
    }
    std::array<bool, batchnorm_output_storage_elements>
        batchnorm_training_output_written{};
    const auto require_batchnorm_training_statistic_close =
        [&](std::string_view name,
            std::size_t channel,
            float actual,
            float expected) {
          const float absolute_error = std::abs(actual - expected);
          const float tolerance = 2.0e-4F + 2.0e-4F * std::abs(expected);
          if (!std::isfinite(actual) || !std::isfinite(expected) ||
              !std::isfinite(absolute_error) || absolute_error > tolerance) {
            throw std::runtime_error(
                "installed Ascend BatchNorm training statistic differs for " +
                std::string(name) + " at channel " +
                std::to_string(channel));
          }
        };
    for (std::size_t channel = 0; channel < batchnorm_channels; ++channel) {
      require_batchnorm_training_statistic_close(
          "mean",
          channel,
          host_batchnorm_training_mean_output[channel],
          host_batchnorm_training_mean_expected[channel]);
      require_batchnorm_training_statistic_close(
          "inverse variance",
          channel,
          host_batchnorm_training_inv_variance_output[channel],
          host_batchnorm_training_inv_variance_expected[channel]);
      require_batchnorm_training_statistic_close(
          "next running mean",
          channel,
          host_batchnorm_training_next_running_mean_output[channel],
          host_batchnorm_training_next_running_mean_expected[channel]);
      require_batchnorm_training_statistic_close(
          "next running variance",
          channel,
          host_batchnorm_training_next_running_variance_output[channel],
          host_batchnorm_training_next_running_variance_expected[channel]);
      for (std::size_t row = 0; row < batchnorm_rows; ++row) {
        const std::size_t physical =
            row * batchnorm_output_row_stride + channel;
        batchnorm_training_output_written[physical] = true;
        const float expected = host_batchnorm_training_expected[physical];
        const float actual = host_batchnorm_training_output[physical];
        const float absolute_error = std::abs(actual - expected);
        const float tolerance = 2.0e-4F + 2.0e-4F * std::abs(expected);
        if (!std::isfinite(actual) || !std::isfinite(expected) ||
            !std::isfinite(absolute_error) || absolute_error > tolerance) {
          throw std::runtime_error(
              "installed Ascend BatchNorm training output differs at "
              "logical index " +
              std::to_string(row * batchnorm_channels + channel));
        }
      }
    }
    for (std::size_t physical = 0;
         physical < host_batchnorm_training_output.size();
         ++physical) {
      if (!batchnorm_training_output_written[physical] &&
          std::bit_cast<std::uint32_t>(
              host_batchnorm_training_output[physical]) !=
              layout_padding_pattern) {
        throw std::runtime_error(
            "installed Ascend BatchNorm training modified output padding at "
            "physical index " +
            std::to_string(physical));
      }
    }
    if (host_batchnorm_training_scale_after !=
            host_batchnorm_training_scale ||
        host_batchnorm_training_bias_after != host_batchnorm_training_bias ||
        host_batchnorm_training_previous_running_mean_after !=
            host_batchnorm_training_previous_running_mean ||
        host_batchnorm_training_previous_running_variance_after !=
            host_batchnorm_training_previous_running_variance) {
      throw std::runtime_error(
          "installed Ascend BatchNorm training graph modified an input");
    }
    for (std::size_t index = 0; index < host_matmul_output.size(); ++index) {
      const float actual = host_matmul_output[index];
      const float expected = host_matmul_expected[index];
      const float absolute_error = std::abs(actual - expected);
      const float tolerance = 1.0e-5F + 1.0e-5F * std::abs(expected);
      if (!std::isfinite(actual) || !std::isfinite(expected) ||
          !std::isfinite(absolute_error) || absolute_error > tolerance) {
        throw std::runtime_error(
            "installed Ascend MatMul output differs at index " +
            std::to_string(index) + ": actual=" +
            std::to_string(actual) + ", expected=" +
            std::to_string(expected));
      }
    }
    if (host_matmul_a_after != host_matmul_a ||
        host_matmul_b_after != host_matmul_b) {
      throw std::runtime_error(
          "installed Ascend MatMul graph modified an input");
    }
    for (std::size_t index = 0; index < host_convolution_output.size();
         ++index) {
      const float actual = host_convolution_output[index];
      const float expected = host_convolution_expected[index];
      const float absolute_error = std::abs(actual - expected);
      const float tolerance = 1.0e-5F + 1.0e-5F * std::abs(expected);
      if (!std::isfinite(actual) || !std::isfinite(expected) ||
          !std::isfinite(absolute_error) || absolute_error > tolerance) {
        throw std::runtime_error(
            "installed Ascend ConvFprop output differs at index " +
            std::to_string(index) + ": actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected));
      }
    }
    if (host_convolution_input_after != host_convolution_input ||
        host_convolution_filter_after != host_convolution_filter) {
      throw std::runtime_error(
          "installed Ascend ConvFprop graph modified an input");
    }
    const std::array<const char*, 6> comparison_names = {
        "CmpEq", "CmpNeq", "CmpGt", "CmpGe", "CmpLt", "CmpLe"};
    const std::array<const std::array<std::uint8_t, 256>*, 6>
        comparison_outputs = {
            &host_comparison_equal_output,
            &host_comparison_not_equal_output,
            &host_comparison_greater_output,
            &host_comparison_greater_equal_output,
            &host_comparison_less_output,
            &host_comparison_less_equal_output};
    const std::array<const std::array<std::uint8_t, 256>*, 6>
        comparison_expected = {
            &host_comparison_equal_expected,
            &host_comparison_not_equal_expected,
            &host_comparison_greater_expected,
            &host_comparison_greater_equal_expected,
            &host_comparison_less_expected,
            &host_comparison_less_equal_expected};
    for (std::size_t operation = 0; operation < comparison_names.size();
         ++operation) {
      for (std::size_t index = 0;
           index < comparison_outputs[operation]->size();
           ++index) {
        const std::uint8_t actual = (*comparison_outputs[operation])[index];
        const std::uint8_t expected = (*comparison_expected[operation])[index];
        if (actual > 1U || actual != expected) {
          throw std::runtime_error(
              std::string("installed Ascend ") +
              comparison_names[operation] +
              " output differs at index " + std::to_string(index) +
              ": actual=" +
              std::to_string(static_cast<unsigned int>(actual)) +
              ", expected=" +
              std::to_string(static_cast<unsigned int>(expected)));
        }
      }
    }
  }
  acl.finish();
}

}  // namespace

int main() {
  try {
    run_pointwise_graph();
    std::cout
        << "PASS installed FlagDNN Graph "
           "Add/Sub/Mul/Div/Min/Max/Identity/Neg/Elu/LeakyReLU/Abs/Ceil/Floor/"
           "ReLU/Reciprocal/Sqrt/Rsqrt/Exp/Log/Tanh/Sigmoid/Softplus/"
           "Swish/Gelu/GeluApproxTanh/Sin/Cos/Tan/Erf/Mod/Pow/"
           "SigmoidBackward/LogicalNot/LogicalAnd/LogicalOr/CmpEq/CmpNeq/"
           "CmpGt/CmpGe/CmpLt/CmpLe/BinarySelect/Reshape/Transpose/Slice/"
           "ReductionSum/ReductionAvg/ReductionMul/BatchNormInference/"
           "RMSNorm/LayerNorm/BatchNormTraining/MatMul/ConvFprop -> NPU "
           "libtriton_jit -> Ascend\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
