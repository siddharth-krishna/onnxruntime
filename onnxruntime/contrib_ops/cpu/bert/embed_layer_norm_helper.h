// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {
namespace contrib {
namespace embed_layer_norm {

Status CheckInputs(const OpKernelContext* context);

Status CheckQuantizedInputs(const OpKernelContext* context);

// TODO(kreeger): move common logic here or inside the base class.
template <typename T>
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
                       std::function<T(T)>* word_embedding_fn = nullptr
                       //F&& word_embeddding_fn,
                       //F&& position_embedding_fn,
                       //F&& segment_embedding_fn,
                       //F&& gamma_fn,
                       //F&& beta_fn,
                       );

}  // namespace embed_layer_norm
}  // namespace contrib
}  // namespace onnxruntime
