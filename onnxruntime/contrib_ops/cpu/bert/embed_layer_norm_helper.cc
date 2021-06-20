// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm_helper.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "onnx/defs/tensor_proto_util.h"

#include "longformer_attention_base.h"

namespace onnxruntime {
namespace contrib {

Status LongformerAttentionBase__CheckInputs(const LongformerAttentionBase* p,
                                            const TensorShape& input_shape,
                                            const TensorShape& weights_shape,
                                            const TensorShape& bias_shape,
                                            const TensorShape& mask_shape,
                                            const TensorShape& global_weights_shape,
                                            const TensorShape& global_bias_shape,
                                            const TensorShape& global_shape) {
  return p->CheckInputs(input_shape, weights_shape, bias_shape, mask_shape, global_weights_shape, global_bias_shape, global_shape);
}

namespace embed_layer_norm {

namespace {

Status CheckInputsInternal(const OpKernelContext* context, int mask_index) {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* segment_ids = context->Input<Tensor>(1);  // optional. nullptr if it's distill-bert
  const Tensor* word_embedding = context->Input<Tensor>(2);
  const Tensor* position_embedding = context->Input<Tensor>(3);
  const Tensor* segment_embedding = context->Input<Tensor>(4);  // optional. nullptr if it's distill-bert
  const Tensor* gamma = context->Input<Tensor>(5);
  const Tensor* beta = context->Input<Tensor>(6);
  const Tensor* mask = context->Input<Tensor>(mask_index);  // optional. nullptr if not provided

  if (nullptr != segment_ids && input_ids->Shape() != segment_ids->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 and 1 shall have same shape");
  }

  if (nullptr != mask && input_ids->Shape() != mask->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 and 7 (mask) shall have same shape");
  }

  const auto& input_dims = input_ids->Shape().GetDims();
  if (input_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input_ids is expected to have 2 dimensions, got ", input_dims.size());
  }

  const auto& word_embedding_dims = word_embedding->Shape().GetDims();
  if (word_embedding_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "word_embedding is expected to have 2 dimensions, got ", word_embedding_dims.size());
  }

  const auto& position_embedding_dims = position_embedding->Shape().GetDims();
  if (position_embedding_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "position_embedding is expected to have 2 dimensions, got ", position_embedding_dims.size());
  }

  if (nullptr != segment_embedding) {
    const auto& segment_embedding_dims = segment_embedding->Shape().GetDims();
    if (segment_embedding_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "segment_embedding is expected to have 2 dimensions, got ", segment_embedding_dims.size());
    }
    if (word_embedding_dims[1] != segment_embedding_dims[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "word_embedding and segment_embedding shall have same dimension 1");
    }
  }

  if (word_embedding_dims[1] != position_embedding_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "word_embedding and position_embedding shall have same dimension 1");
  }

  const auto& beta_dims = beta->Shape().GetDims();
  if (beta_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "beta is expected to have 1 dimensions, got ", beta_dims.size());
  }

  if (beta_dims[0] != word_embedding_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "beta is expected to have size of ", word_embedding_dims[1], ", got ", beta_dims[0]);
  }

  const auto& gamma_dims = gamma->Shape().GetDims();
  if (gamma_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have 1 dimensions, got ", gamma_dims.size());
  }

  if (gamma_dims[0] != word_embedding_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have size of ", word_embedding_dims[1], ", got ", gamma_dims[0]);
  }

  return Status::OK();
}

}  // namespace

Status CheckInputs(const OpKernelContext* context) {
  return CheckInputsInternal(context, /*mask_index=*/7);
}

Status CheckQuantizedInputs(const OpKernelContext* context) {
  // Optional mask index is the last input after quantization values:
  ORT_RETURN_IF_ERROR(CheckInputsInternal(context, /*mask_index=*/17));

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(7)),
      "Word embedding scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(8)),
      "Position embedding scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(9)),
      "Segment embedding scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(10)),
      "Layer norm weights scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(11)),
      "Layer norm bias must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(12)),
      "Word embedding zero point must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(13)),
      "Position embedding zero point must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(14)),
      "Segment embedding zero point must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(15)),
      "Layer norm weights zero point must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(16)),
      "Layer norm bias zero point must be a scalar or 1D tensor of size 1");

  return Status::OK();
}

template <typename T, typename F>
Status ComputeInternal(int batch_size,
                       int sequence_size,
                       int64_t hidden_size,
                       const int32_t* input_ids_data,
                       const int32_t* segment_ids_data,
                       const T* word_embedding_data,
                       const T* position_embedding_data,
                       const T* segment_embedding_data,
                       const T* gamma_data,
                       const T* beta_data,
                       T* output_data,
                       std::function<T(T)>* word_embedding_fn
                       //F&& position_embedding_fn,
                       //F&& segment_embedding_fn,
                       //F&& gamma_fn,
                       //F&& beta_fn,
                       ) {
  //
  // TODO(kreeger): write me.
  //
  {
    std::atomic_bool failed{false};

    int n = batch_size * sequence_length;
    concurrency::ThreadPool::TryBatchParallelFor(
        context->GetOperatorThreadPool(), n, [=, &failed](ptrdiff_t index) {
      int word_col_index = input_ids_data[index];
      if (word_col_index < 0 || word_col_index >= word_embedding_length) {
        failed.store(true, std::memory_order_release);
        return;
      }
      int position_col_index = index % sequence_length;
      if (position_col_index >= position_embedding_length) {
        failed.store(true, std::memory_order_release);
        return;
      }
      int segment_col_index = 0;
      if (nullptr != segment_ids_data) {
        segment_col_index = segment_ids_data[index];
        if (segment_col_index < 0 || segment_col_index >= segment_embedding_length) {
          failed.store(true, std::memory_order_release);
          return;
        }
      }

      // Grab inputs for the embeddings for the current batch index:
      // TODO - will need type annontation on these instead of uint8_t
      const uint8_t* input_word_embedding = word_embedding_data + (word_col_index * hidden_size);
      const uint8_t* input_position_embedding =
          position_embedding_data + (position_col_index * hidden_size);
      const uint8_t* input_segment_embedding = nullptr;
      if (segment_embedding_data != nullptr) {
        input_segment_embedding = segment_embedding_data + (segment_col_index * hidden_size);
      }

      T* output = output_data + (index * hidden_size);

      T sum = static_cast<T>(0);
      for (int i = 0; i < hidden_size; ++i) {
        // pass a lambda for these dequantize calls.
        T cur_word_embedding = word_embedding_fn ==
                                       nullptr
                                   ? input_word_embedding[i]
                                   : word_embedding_fn(input_word_embedding[i]);
        T subtotal = word_embeddding_fn(i) + position_embedding_fn(i);
        if (segment_embedding_data != nullptr) {
          subtotal += segment_embedding_fn(i);
        }
        output[i] = subtotal;
        sum += subtotal;
      }

      T mean = sum / hidden_size;
      sum = 0;

      for (int i = 0; i < hidden_size; i++) {
        T a = output[i] - mean;
        output[i] = a;
        sum += a * a;
      }

      T e = sqrt(sum / hidden_size + static_cast<T>(epsilon_));
      for (int i = 0; i < hidden_size; i++) {
        T cur_weight = gamma_fn(i);
        T cur_bias = beta_fn(i);
        output[i] = output[i] / e * cur_weight + cur_bias;
      }
    }, 0);

    if (failed.load(std::memory_order_acquire)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input index out of range");
    }
  }

  // Calculate mask
  if (nullptr != mask) {
    const int32_t* mask_data = mask->template Data<int32_t>();
    for (int b = 0; b < batch_size; b++) {
      // TODO(kreeger): Fix static cast warning here:
      mask_index->template MutableData<int32_t>()[b] =
          static_cast<int32_t>(std::count_if(mask_data + (b * sequence_length),
                                             mask_data + (b * sequence_length) + sequence_length,
                                             [](int v) { return v == 1; }));
    }
  } else {
    memset(mask_index->template MutableData<int32_t>(), 0, batch_size * sizeof(int32_t));
  }

  return Status::OK();

  return Status::OK();
}

}  // namespace embed_layer_norm
}  // namespace contrib
}  // namespace onnxruntime
