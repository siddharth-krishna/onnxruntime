// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qembed_layer_norm.h"

#include "embed_layer_norm_helper.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

namespace {

template <typename T>
Status GetQuantizedInputTensorValue(OpKernelContext* context, int index, T& value) {
  const Tensor* tensor = context->Input<Tensor>(index);
  // TODO(kreeger): Consider moving this to the ::CheckQuantizedInputs() method.
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(tensor));
  value = *(tensor->template Data<T>());
  return Status::OK();
}

}  // namespace

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
  // TODO(kreeger): Get |epsilon| attribute here.
}

template <typename T>
Status QEmbedLayerNorm<T>::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(embed_layer_norm::CheckQuantizedInputs(context));

  // TODO(kreeger): Move verbose descriptions here.
  /*
  Input Tensors List:
  [0] input_ids
  [1] segment_ids
  [2] word_embedding_quant
  [3] word_embedding_scale
  [4] word_embedding_zero_point
  [5] position_embedding_quant
  [6] position_embedding_scale
  [7] position_embedding_zero_point
  [8] segment_embedding_quant
  [9] segment_embedding_scale
  [10] segment_embedding_zero_point
  [11] layer_norm_weight_quant
  [12] layer_norm_weight_scale
  [13] layer_norm_weight_zero_point
  [14] layer_norm_bias_quant
  [15] layer_norm_bias_scale
  [16] layer_norm_bias_zero_point
  */
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* segment_ids = context->Input<Tensor>(1);
  const Tensor* word_embedding = context->Input<Tensor>(3);
  const Tensor* position_embedding = context->Input<Tensor>(5);
  const Tensor* segment_embedding = context->Input<Tensor>(8);
  const Tensor* layer_norm_weight = context->Input<Tensor>(11);
  const Tensor* layer_norm_bias = context->Input<Tensor>(14);

  //
  // TODO(kreeger): MOVE QUANT INPUTS TO THE END. ALLOW FOR MORE RE-USE OF KERNEL API?
  //

  // TODO(kreeger): Move verbose descriptions here.
  /*
  Output Tensors List:
  [0] layernorm_out
  [1] mask_index_out
  */
  const auto& input_dims = input_ids->Shape().GetDims();
  int64_t hidden_size = word_embedding->Shape()[1];

  TensorShape output_shape({input_dims[0], input_dims[1], hidden_size});
  Tensor* output = context->Output(0, output_shape);

  TensorShape mask_index_shape({input_dims[0]});
  Tensor* mask_index = context->Output(1, mask_index_shape);

  //
  // TODO(kreeger): write me.
  //

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
