// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qembed_layer_norm.h"

namespace onnxruntime {
namespace contrib {

// TODO - op registration goes here.

QEmbedLayerNorm::QEmbedLayerNorm(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info) {
  // TOOD - anything needed here?
}

Status QEmbedLayerNorm::Compute(OpKernelContext* context) const {
  if (context != nullptr) {
    // compiler foo
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
