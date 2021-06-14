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

  /*
  Tensors List:
  [0] input_ids
  [1] segmend_ids
  [2] word_embedding_quant
  [3] word_embedding_scale
  [4] word_embedding_zp
  [5] position_embedding_quant
  [6] position_embedding_scale
  [7] position_embedding_zp
  [8] segment_embedding_quant
  [9] segment_embedding_scale
  [10] segment_embedding_zp
  [11] gamma (quant/supprt?)
  [12] beta (quant/support?)
  [13] mask (quant/support?)
  */

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
