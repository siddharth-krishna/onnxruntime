// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qembed_layer_norm.h"

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// This op is internal-only, so register outside of onnx:
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      QEmbedLayerNormalization,                                    \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      QEmbedLayerNorm<T>);

REGISTER_KERNEL_TYPED(float)


template <typename T>
QEmbedLayerNorm<T>::QEmbedLayerNorm(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info) {
  // TOOD - anything needed here?
}

template <typename T>
Status QEmbedLayerNorm<T>::Compute(OpKernelContext* context) const {
  if (context != nullptr) {
    // compiler foo
  }



  //
  // TODO(kreeger): LEFT OFF RIGHT HERE.
  // NEED TO GENERATE SOME TEST DATA IN PYTHON TO ACTUALLY RUN THIS MODEL.
  // MIGHT CONSIDER JUST WRITING AN OP UNIT TEST FIRST.
  //




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
