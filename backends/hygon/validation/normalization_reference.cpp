/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "normalization_reference.hpp"

#include "hip_driver.hpp"
#include "tensor_io.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::validation::hygon {
namespace {

constexpr std::size_t kX = 0;
constexpr std::size_t kScale = 1;
constexpr std::size_t kBias = 2;
constexpr std::size_t kPreviousMean = 3;
constexpr std::size_t kPreviousVariance = 4;
constexpr std::size_t kY = 5;
constexpr std::size_t kSavedMean = 6;
constexpr std::size_t kSavedInvVariance = 7;
constexpr std::size_t kNextMean = 8;
constexpr std::size_t kNextVariance = 9;
constexpr std::size_t kBatchnormTensorCount = 10;

std::vector<int> checked_ints(std::span<const std::int64_t> values,
                              std::string_view role) {
  std::vector<int> result;
  result.reserve(values.size());
  for (const std::int64_t value : values) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(role) +
                                  " does not fit hipDNN int metadata");
    }
    result.push_back(static_cast<int>(value));
  }
  return result;
}

std::vector<std::int64_t>
dense_strides(std::span<const std::int64_t> dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::invalid_argument("normalization dimensions overflow");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

bool same_descriptor(const ReferenceTensor &left,
                     const ReferenceTensor &right) {
  return left.data_type == right.data_type &&
         left.dimensions == right.dimensions && left.strides == right.strides;
}

std::size_t tensor_bytes(const ReferenceTensor &tensor) {
  const std::size_t count = tensor_io::storage_element_count(tensor);
  const std::size_t element_size = tensor_io::data_type_size(tensor.data_type);
  if (count > std::numeric_limits<std::size_t>::max() / element_size) {
    throw std::overflow_error("normalization tensor byte size overflows");
  }
  return count * element_size;
}

void set_tensor_descriptor(hipdnnTensorDescriptor_t descriptor,
                           const ReferenceTensor &tensor) {
  const std::vector<int> dimensions =
      checked_ints(tensor.dimensions, "tensor dimensions");
  const std::vector<int> strides =
      checked_ints(tensor.strides, "tensor strides");
  check_hipdnn_status(
      hipdnnSetTensorNdDescriptor(descriptor, HIPDNN_DATA_FLOAT,
                                  static_cast<int>(dimensions.size()),
                                  dimensions.data(), strides.data()),
      "hipdnnSetTensorNdDescriptor(BatchNorm)");
}

void *binding_pointer(std::span<const flagdnnBinding_t> bindings,
                      const ReferenceTensor &tensor) {
  const auto found = std::find_if(bindings.begin(), bindings.end(),
                                  [&](const flagdnnBinding_t &binding) {
                                    return binding.uid == tensor.uid;
                                  });
  if (found == bindings.end() || found->device_pointer == nullptr) {
    throw std::invalid_argument("hipDNN BatchNorm binding is missing");
  }
  return static_cast<void *>(
      static_cast<std::uint8_t *>(found->device_pointer) +
      tensor.binding_byte_offset);
}

HipdnnCapability
structural_capability(const HipdnnNormalizationOperation &operation,
                      std::span<const ReferenceTensor> tensors) {
  if (operation.kind == HipdnnNormalizationKind::kUnavailable) {
    return HipdnnCapability::vendor_unsupported(
        operation.unavailable_reason.empty()
            ? "no exact hipDNN normalization primitive"
            : operation.unavailable_reason);
  }
  if (operation.kind != HipdnnNormalizationKind::kBatchnormTraining) {
    return HipdnnCapability::invalid_adapter_contract(
        "unknown hipDNN normalization operation");
  }
  if (tensors.size() != kBatchnormTensorCount) {
    return HipdnnCapability::invalid_adapter_contract(
        "hipDNN BatchNorm training requires exactly ten tensors");
  }
  if (!std::isfinite(operation.epsilon)) {
    return HipdnnCapability::invalid_adapter_contract(
        "BatchNorm epsilon must be finite");
  }
  if (operation.epsilon < HIPDNN_BN_MIN_EPSILON) {
    return HipdnnCapability::vendor_unsupported(
        "epsilon is below the hipDNN BatchNorm minimum");
  }
  if (!std::isfinite(operation.momentum) || operation.momentum < 0.0 ||
      operation.momentum > 1.0) {
    return HipdnnCapability::invalid_adapter_contract(
        "momentum is outside [0, 1]");
  }

  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const ReferenceTensor &tensor = tensors[index];
    if (tensor.uid <= 0 || tensor.dimensions.size() != tensor.strides.size()) {
      return HipdnnCapability::invalid_adapter_contract(
          "BatchNorm tensor UID or dimensions/strides metadata is invalid");
    }
    if (tensor.data_type != FLAGDNN_DATA_FLOAT32 ||
        tensor.dimensions.size() != 4) {
      return HipdnnCapability::vendor_unsupported(
          "only rank-four FP32 tensors have an exact hipDNN mapping");
    }
    for (std::size_t other = 0; other < index; ++other) {
      if (tensor.uid == tensors[other].uid) {
        return HipdnnCapability::invalid_adapter_contract(
            "BatchNorm tensor UIDs must be distinct");
      }
    }
    if (tensor.binding_byte_offset % sizeof(float) != 0) {
      return HipdnnCapability::invalid_adapter_contract(
          "BatchNorm binding byte offset is not FP32 aligned");
    }
    for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
      if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
        return HipdnnCapability::invalid_adapter_contract(
            "BatchNorm tensor dimensions/strides must be positive");
      }
      if (tensor.dimensions[axis] > std::numeric_limits<int>::max() ||
          tensor.strides[axis] > std::numeric_limits<int>::max()) {
        return HipdnnCapability::vendor_unsupported(
            "BatchNorm tensor metadata does not fit hipDNN int descriptors");
      }
    }
    try {
      (void)tensor_bytes(tensor);
    } catch (const std::overflow_error &error) {
      return HipdnnCapability::vendor_unsupported(error.what());
    }
  }

  const ReferenceTensor &x = tensors[kX];
  const ReferenceTensor &y = tensors[kY];
  if (x.dimensions != y.dimensions ||
      x.strides != dense_strides(x.dimensions) ||
      y.strides != dense_strides(y.dimensions)) {
    return HipdnnCapability::vendor_unsupported(
        "hipDNN BatchNorm X/Y must use dense NCHW storage");
  }
  const std::vector<std::int64_t> parameters = {1, x.dimensions[1], 1, 1};
  if (tensors[kScale].dimensions != parameters ||
      tensors[kScale].strides != dense_strides(parameters)) {
    return HipdnnCapability::invalid_adapter_contract(
        "BatchNorm parameters must be dense [1,C,1,1]");
  }
  for (const std::size_t index :
       {kBias, kPreviousMean, kPreviousVariance, kSavedMean, kSavedInvVariance,
        kNextMean, kNextVariance}) {
    if (!same_descriptor(tensors[kScale], tensors[index])) {
      return HipdnnCapability::invalid_adapter_contract(
          "hipDNN adapter requires one descriptor for scale/bias/statistics");
    }
  }
  const std::int64_t batch = x.dimensions[0];
  const std::int64_t height = x.dimensions[2];
  const std::int64_t width = x.dimensions[3];
  if (batch > std::numeric_limits<std::int64_t>::max() / height ||
      batch * height > std::numeric_limits<std::int64_t>::max() / width ||
      batch * height * width <= 1) {
    return HipdnnCapability::vendor_unsupported(
        "BatchNorm reduction extent must be representable and exceed one");
  }
  return {};
}

class TrainingResources final {
public:
  explicit TrainingResources(std::span<const ReferenceTensor> tensors) {
    check_hipdnn_status(hipdnnCreate(&handle_), "hipdnnCreate(BatchNorm)");
    try {
      check_hipdnn_status(hipdnnCreateTensorDescriptor(&x_descriptor_),
                          "hipdnnCreateTensorDescriptor(X)");
      check_hipdnn_status(hipdnnCreateTensorDescriptor(&y_descriptor_),
                          "hipdnnCreateTensorDescriptor(Y)");
      check_hipdnn_status(hipdnnCreateTensorDescriptor(&parameter_descriptor_),
                          "hipdnnCreateTensorDescriptor(parameters)");
      set_tensor_descriptor(x_descriptor_, tensors[kX]);
      set_tensor_descriptor(y_descriptor_, tensors[kY]);
      set_tensor_descriptor(parameter_descriptor_, tensors[kScale]);
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~TrainingResources() { cleanup(); }

  TrainingResources(const TrainingResources &) = delete;
  TrainingResources &operator=(const TrainingResources &) = delete;

  void execute(const HipdnnNormalizationOperation &operation, void *x,
               void *scale, void *bias, void *y, void *saved_mean,
               void *saved_inv_variance, void *next_mean, void *next_variance,
               hipStream_t stream) {
    check_hipdnn_status(hipdnnSetStream(handle_, stream),
                        "hipdnnSetStream(BatchNorm)");
    float alpha = 1.0F;
    float beta = 0.0F;
    check_hipdnn_status(hipdnnBatchNormalizationForwardTraining(
                            handle_, HIPDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                            x_descriptor_, x, y_descriptor_, y,
                            parameter_descriptor_, scale, bias,
                            operation.momentum, next_mean, next_variance,
                            operation.epsilon, saved_mean, saved_inv_variance),
                        "hipdnnBatchNormalizationForwardTraining");
  }

private:
  void cleanup() noexcept {
    if (parameter_descriptor_ != nullptr) {
      (void)hipdnnDestroyTensorDescriptor(parameter_descriptor_);
      parameter_descriptor_ = nullptr;
    }
    if (y_descriptor_ != nullptr) {
      (void)hipdnnDestroyTensorDescriptor(y_descriptor_);
      y_descriptor_ = nullptr;
    }
    if (x_descriptor_ != nullptr) {
      (void)hipdnnDestroyTensorDescriptor(x_descriptor_);
      x_descriptor_ = nullptr;
    }
    if (handle_ != nullptr) {
      (void)hipdnnDestroy(handle_);
      handle_ = nullptr;
    }
  }

  hipdnnHandle_t handle_ = nullptr;
  hipdnnTensorDescriptor_t x_descriptor_ = nullptr;
  hipdnnTensorDescriptor_t y_descriptor_ = nullptr;
  hipdnnTensorDescriptor_t parameter_descriptor_ = nullptr;
};

void real_execute_gate(const HipdnnNormalizationOperation &operation,
                       std::span<const ReferenceTensor> tensors) {
  TrainingResources resources(tensors);
  Stream stream;
  std::vector<std::unique_ptr<DeviceBuffer>> buffers;
  buffers.reserve(tensors.size());
  for (const ReferenceTensor &tensor : tensors) {
    const std::size_t bytes = tensor.binding_byte_offset + tensor_bytes(tensor);
    auto buffer = std::make_unique<DeviceBuffer>(bytes);
    check_hip(hipMemsetAsync(buffer->opaque(), 0, bytes, stream.get()),
              "hipMemsetAsync(BatchNorm gate)");
    buffers.push_back(std::move(buffer));
  }
  const auto pointer = [&](std::size_t index) {
    return buffers[index]->opaque_at(tensors[index].binding_byte_offset);
  };
  check_hip(hipMemcpyAsync(pointer(kNextMean), pointer(kPreviousMean),
                           tensor_bytes(tensors[kPreviousMean]),
                           hipMemcpyDeviceToDevice, stream.get()),
            "hipMemcpyAsync(BatchNorm gate mean)");
  check_hip(hipMemcpyAsync(pointer(kNextVariance), pointer(kPreviousVariance),
                           tensor_bytes(tensors[kPreviousVariance]),
                           hipMemcpyDeviceToDevice, stream.get()),
            "hipMemcpyAsync(BatchNorm gate variance)");
  resources.execute(operation, pointer(kX), pointer(kScale), pointer(kBias),
                    pointer(kY), pointer(kSavedMean),
                    pointer(kSavedInvVariance), pointer(kNextMean),
                    pointer(kNextVariance), stream.get());
  stream.synchronize();
}

} // namespace

HipdnnNormalizationOperation
make_hipdnn_batchnorm_training_operation(double epsilon, double momentum) {
  HipdnnNormalizationOperation result;
  result.kind = HipdnnNormalizationKind::kBatchnormTraining;
  result.epsilon = epsilon;
  result.momentum = momentum;
  return result;
}

HipdnnNormalizationOperation
make_hipdnn_normalization_unavailable(std::string reason) {
  HipdnnNormalizationOperation result;
  result.kind = HipdnnNormalizationKind::kUnavailable;
  result.unavailable_reason = std::move(reason);
  return result;
}

HipdnnCapability
hipdnn_normalization_capability(const HipdnnNormalizationOperation &operation,
                                std::span<const ReferenceTensor> tensors) {
  const HipdnnCapability structural = structural_capability(operation, tensors);
  if (!structural.supported) {
    return structural;
  }
  try {
    real_execute_gate(operation, tensors);
  } catch (const HipdnnStatusError &error) {
    if (!hipdnn_status_is_capability(error.status())) {
      throw;
    }
    return HipdnnCapability::vendor_unsupported(
        std::string("hipDNN BatchNorm runtime capability: ") + error.what());
  }
  return {};
}

std::string_view
hipdnn_normalization_kind_name(HipdnnNormalizationKind kind) noexcept {
  switch (kind) {
  case HipdnnNormalizationKind::kBatchnormTraining:
    return "batchnorm";
  case HipdnnNormalizationKind::kUnavailable:
    return "normalization";
  }
  return "normalization";
}

ReferenceTensor
batchnorm_dense_reference_tensor(const ReferenceTensor &tensor) {
  if (tensor.dimensions.size() != 4 || tensor.strides.size() != 4) {
    throw std::invalid_argument(
        "hipDNN BatchNorm reference requires rank-four X/Y tensors");
  }
  ReferenceTensor result = tensor;
  result.strides = dense_strides(result.dimensions);
  result.binding_byte_offset = 0;
  return result;
}

class HipdnnNormalizationPlan::Impl final {
public:
  Impl(HipdnnNormalizationOperation operation,
       std::vector<ReferenceTensor> tensors)
      : operation_(std::move(operation)), tensors_(std::move(tensors)),
        resources_(checked_tensors(operation_, tensors_)) {}

  [[nodiscard]] std::size_t workspace_size() const noexcept { return 0; }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) {
    (void)workspace;
    (void)workspace_size;
    hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
    void *previous_mean = binding_pointer(bindings, tensors_[kPreviousMean]);
    void *previous_variance =
        binding_pointer(bindings, tensors_[kPreviousVariance]);
    void *next_mean = binding_pointer(bindings, tensors_[kNextMean]);
    void *next_variance = binding_pointer(bindings, tensors_[kNextVariance]);
    check_hip(hipMemcpyAsync(next_mean, previous_mean,
                             tensor_bytes(tensors_[kPreviousMean]),
                             hipMemcpyDeviceToDevice, hip_stream),
              "hipMemcpyAsync(BatchNorm running mean)");
    check_hip(hipMemcpyAsync(next_variance, previous_variance,
                             tensor_bytes(tensors_[kPreviousVariance]),
                             hipMemcpyDeviceToDevice, hip_stream),
              "hipMemcpyAsync(BatchNorm running variance)");
    resources_.execute(operation_, binding_pointer(bindings, tensors_[kX]),
                       binding_pointer(bindings, tensors_[kScale]),
                       binding_pointer(bindings, tensors_[kBias]),
                       binding_pointer(bindings, tensors_[kY]),
                       binding_pointer(bindings, tensors_[kSavedMean]),
                       binding_pointer(bindings, tensors_[kSavedInvVariance]),
                       next_mean, next_variance, hip_stream);
  }

private:
  static const std::vector<ReferenceTensor> &
  checked_tensors(const HipdnnNormalizationOperation &operation,
                  const std::vector<ReferenceTensor> &tensors) {
    const HipdnnCapability capability =
        structural_capability(operation, tensors);
    if (!capability.supported) {
      throw std::invalid_argument("invalid hipDNN normalization plan: " +
                                  capability.reason);
    }
    return tensors;
  }

  HipdnnNormalizationOperation operation_;
  std::vector<ReferenceTensor> tensors_;
  TrainingResources resources_;
};

HipdnnNormalizationPlan::HipdnnNormalizationPlan(
    HipdnnNormalizationOperation operation,
    std::vector<ReferenceTensor> tensors)
    : impl_(std::make_unique<Impl>(std::move(operation), std::move(tensors))) {}

HipdnnNormalizationPlan::~HipdnnNormalizationPlan() = default;
HipdnnNormalizationPlan::HipdnnNormalizationPlan(
    HipdnnNormalizationPlan &&) noexcept = default;
HipdnnNormalizationPlan &HipdnnNormalizationPlan::operator=(
    HipdnnNormalizationPlan &&) noexcept = default;

std::size_t HipdnnNormalizationPlan::workspace_size() const noexcept {
  return impl_->workspace_size();
}

void HipdnnNormalizationPlan::execute(
    std::span<const flagdnnBinding_t> bindings, void *workspace,
    std::size_t workspace_size, flagdnnStream_t stream) {
  impl_->execute(bindings, workspace, workspace_size, stream);
}

} // namespace flagdnn::validation::hygon
