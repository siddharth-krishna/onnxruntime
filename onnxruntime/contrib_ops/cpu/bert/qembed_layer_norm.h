// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// Quantized version of QEmbedLayerNorm.
// TODO(kreeger): does this need to be uint8 and/or int8?
template <typename T>
class QEmbedLayerNorm final : public OpKernel {
 public:
  explicit QEmbedLayerNorm(const OpKernelInfo& op_kernel_info);

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
