// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/quantization_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

static void RunTest(
    const std::vector<int32_t>& input_ids_data,
    const std::vector<int32_t>& segment_ids_data,
    const std::vector<float>& word_embedding_data,
    const std::vector<float>& position_embedding_data,
    const std::vector<float>& segment_embedding_data,
    const std::vector<float>& output_data,
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

  std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

  // Input and output shapes
  //   Input 0 - input_ids                 : (batch_size, sequence_length)
  //   Input 1 - segment_ids               : (batch_size, sequence_length)
  //   Input 2 - word_embedding_quant      : (,hidden_size)
  //   Input 3 - word_embedding_scale      : (float32)
  //   Input 4 - word_embedding_zero_point : (uint8_t)
  // 
  //   Input 3 - position_embedding : (,hidden_size)
  //   Input 4 - segment_embedding  : (,hidden_size)
  //   Input 5 - gamma              : (hidden_size)
  //   Input 6 - beta               : (hidden_size)
  //   Input 7 - mask               : (batch_size, sequence_length)
  //   Output 0 - output            : (batch_size, sequence_length, hidden_size)
  //   Output 1 - mask_index        : (batch_size)

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

  OpTester tester("QEmbedLayerNormalization", 1, onnxruntime::kMSDomain);

  // Operator inputs passed in at int32_t:
  tester.AddInput<int32_t>("input_ids", input_ids_dims, input_ids_data);
  tester.AddInput<int32_t>("segment_ids", segment_ids_dims, segment_ids_data);

  // Quantized uint8_t initializer inputs:
  tester.AddInput<uint8_t>("word_embedding_data",
                           word_embedding_dims,
                           word_embedding_data_quant);
  tester.AddInput<float>("word_embedding_scale", {1}, {word_embedding_scale});
  tester.AddInput<uint8_t>("word_embedding_zero_point", {1}, {word_embedding_zero_point});

  tester.AddInput<uint8_t>("position_embedding_data",
                           position_embedding_dims,
                           position_embedding_data_quant);
  tester.AddInput<float>("position_embedding_scale", {1}, {position_embedding_scale});
  tester.AddInput<uint8_t>("position_embedding_zero_point", {1}, {position_embedding_zero_point});

  tester.AddInput<uint8_t>("segment_embedding_data",
                           segment_embedding_dims,
                           segment_embedding_data_quant);
  tester.AddInput<float>("segment_embedding_scale", {1}, {segment_embedding_scale});
  tester.AddInput<uint8_t>("segment_embedding_zero_point", {1}, {segment_embedding_zero_point});

  //
  // TODO(kreeger): Left off right here - need to the the weights and bias + the additional output!
  //

  // Outputs:
  tester.AddOutput<float>("output", output_dims, output_data);

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

  std::vector<float> output_data = {
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337};

  RunTest(input_ids_data,
          segment_ids_data,
          word_embedding_data,
          position_embedding_data,
          segment_embedding_data,
          output_data,
          batch_size,
          sequence_length,
          hidden_size);
}  // namespace test

}  // namespace test
}  // namespace onnxruntime
