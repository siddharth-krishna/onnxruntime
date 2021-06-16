// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/quantization_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr float kEpsilon = 1e-12f;

static void RunTest(
    const std::vector<int32_t>& input_ids_data,
    const std::vector<int32_t>& segment_ids_data,
    const std::vector<float>& word_embedding_data,
    const std::vector<float>& position_embedding_data,
    const std::vector<float>& segment_embedding_data,
    const std::vector<float>& layer_norm_weight_data,
    const std::vector<float>& layer_norm_bias_data,
    const std::vector<float>& output_data,
    const std::vector<int32_t>& mask_index_data,
    int batch_size,
    int sequence_length,
    int hidden_size) {
  ASSERT_TRUE(word_embedding_data.size() % hidden_size == 0);
  ASSERT_TRUE(position_embedding_data.size() % hidden_size == 0);
  ASSERT_TRUE(segment_embedding_data.size() % hidden_size == 0);

  std::vector<int64_t> input_ids_dims = {batch_size, sequence_length};
  std::vector<int64_t> segment_ids_dims = {batch_size, sequence_length};
  std::vector<int64_t> word_embedding_dims =
      {static_cast<int64_t>(word_embedding_data.size() / hidden_size), hidden_size};
  std::vector<int64_t> position_embedding_dims =
      {static_cast<int64_t>(position_embedding_data.size() / hidden_size), hidden_size};
  std::vector<int64_t> segment_embedding_dims =
      {static_cast<int64_t>(segment_embedding_data.size() / hidden_size), hidden_size};
  std::vector<int64_t> layer_norm_weight_dims = {hidden_size};
  std::vector<int64_t> layer_norm_bias_dims = {hidden_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> mask_index_dims = {batch_size};

  // TODO(kreeger): Document input/output matrix values here!

  float word_embedding_scale = 0.0f;
  uint8_t word_embedding_zero_point = 0;
  std::vector<uint8_t> word_embedding_data_quant = QuantizeLinear<uint8_t, /*symmetric=*/false>(
      word_embedding_data, word_embedding_scale, word_embedding_zero_point);

  float position_embedding_scale = 0.0f;
  uint8_t position_embedding_zero_point = 0;
  std::vector<uint8_t> position_embedding_data_quant = QuantizeLinear<uint8_t, /*symmetric=*/false>(
      position_embedding_data, position_embedding_scale, position_embedding_zero_point);

  float segment_embedding_scale = 0.0f;
  uint8_t segment_embedding_zero_point = 0;
  std::vector<uint8_t> segment_embedding_data_quant = QuantizeLinear<uint8_t, /*symmetric=*/false>(
      segment_embedding_data, segment_embedding_scale, segment_embedding_zero_point);

  float layer_norm_weight_scale = 0.0f;
  uint8_t layer_norm_weight_zero_point = 0;
  std::vector<uint8_t> layer_norm_weight_data_quant = QuantizeLinear<uint8_t, /*symmetric=*/false>(
      layer_norm_weight_data, layer_norm_weight_scale, layer_norm_weight_zero_point);

  float layer_norm_bias_scale = 0.0f;
  uint8_t layer_norm_bias_zero_point = 0;
  std::vector<uint8_t> layer_norm_bias_data_quant = QuantizeLinear<uint8_t, /*symmetric=*/false>(
      layer_norm_bias_data, layer_norm_bias_scale, layer_norm_bias_zero_point);

  OpTester tester("QEmbedLayerNormalization", 1, onnxruntime::kMSDomain);

  // Operator inputs passed in at int32_t:
  tester.AddInput<int32_t>("input_ids", input_ids_dims, input_ids_data);
  tester.AddInput<int32_t>("segment_ids", segment_ids_dims, segment_ids_data);

  // Quantized initializer inputs:
  tester.AddInput<uint8_t>("word_embedding_data",
                           word_embedding_dims,
                           word_embedding_data_quant);
  tester.AddInput<uint8_t>("position_embedding_data",
                           position_embedding_dims,
                           position_embedding_data_quant);
  tester.AddInput<uint8_t>("segment_embedding_data",
                           segment_embedding_dims,
                           segment_embedding_data_quant);
  tester.AddInput<uint8_t>("layer_norm_weight",
                           layer_norm_weight_dims,
                           layer_norm_weight_data_quant);
  tester.AddInput<uint8_t>("layer_norm_bias",
                           layer_norm_bias_dims,
                           layer_norm_bias_data_quant);

  // Quantized scales:
  tester.AddInput<float>("word_embedding_scale", {1}, {word_embedding_scale});
  tester.AddInput<float>("position_embedding_scale", {1}, {position_embedding_scale});
  tester.AddInput<float>("segment_embedding_scale", {1}, {segment_embedding_scale});
  tester.AddInput<float>("layer_norm_weight_scale", {1}, {layer_norm_weight_scale});
  tester.AddInput<float>("layer_norm_bias_scale", {1}, {layer_norm_bias_scale});

  // Quantized zero points:
  tester.AddInput<uint8_t>("word_embedding_zero_point", {1}, {word_embedding_zero_point});
  tester.AddInput<uint8_t>("position_embedding_zero_point", {1}, {position_embedding_zero_point});
  tester.AddInput<uint8_t>("segment_embedding_zero_point", {1}, {segment_embedding_zero_point});
  tester.AddInput<uint8_t>("layer_norm_weight_zero_point", {1}, {layer_norm_weight_zero_point});
  tester.AddInput<uint8_t>("layer_norm_bias_zero_point", {1}, {layer_norm_bias_zero_point});

  // TODO(kreeger): Add optional mask arg here! 

  // Outputs:
  tester.AddOutput<float>("output", output_dims, output_data);
  tester.AddOutput<int32_t>("mask_index", mask_index_dims, mask_index_data);

  // Attributes:
  tester.AddAttribute("epsilon", kEpsilon);

  tester.Run();
}

}  // namespace

TEST(QEmbedLayerNormTest, Shim) {
  int something = 1;
  ASSERT_TRUE(something > 0);

  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  // TODO(kreeger): refactor this stuff with the values in the f32 test.
  std::vector<int32_t> input_ids_data = {
      1, 3};

  std::vector<int32_t> segment_ids_data = {
      0, 1};

  std::vector<float> word_embedding_data = {
      0.2f, 0.1f, 0.4f, -0.6f,
      0.3f, 0.2f, 0.5f, 0.6f,
      0.8f, 0.6f, 0.9f, 1.2f,
      0.1f, 0.3f, 0.5f, 0.9f,
      1.0f, -2.0f, 1.1f, 0.8f};

  std::vector<float> position_embedding_data = {
      0.1f, 0.1f, 0.4f, 0.6f,
      0.6f, 0.0f, 0.8f, 0.6f,
      0.3f, 0.9f, -2.0f, 0.8f};

  std::vector<float> segment_embedding_data = {
      0.3f, 0.4f, 0.9f, 0.1f,
      0.7f, 0.3f, 0.5f, 0.2f};

  std::vector<float> layer_norm_weight_data = {
      0.25f, 0.15f, 0.45f, -0.66f};

  std::vector<float> layer_norm_bias_data = {
      0.6f, 0.2f, 0.5f, -0.6f};

  std::vector<float> output_data = {
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337};

  std::vector<int32_t> mask_index_data = {
      2};

  RunTest(input_ids_data,
          segment_ids_data,
          word_embedding_data,
          position_embedding_data,
          segment_embedding_data,
          layer_norm_weight_data,
          layer_norm_bias_data,
          output_data,
          mask_index_data,
          batch_size,
          sequence_length,
          hidden_size);
}  // namespace test

}  // namespace test
}  // namespace onnxruntime
