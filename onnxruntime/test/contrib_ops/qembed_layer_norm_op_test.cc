// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

static void RunTest(
    const std::vector<int32_t>& input_ids_data,
    const std::vector<int32_t>& segment_ids_data,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int hidden_size) {
  std::vector<int64_t> input_ids_dims = {batch_size, sequence_length};
  std::vector<int64_t> segment_ids_dims = {batch_size, sequence_length};

  std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

  // TODO - update this documentation here!
  // Input and output shapes
  //   Input 0 - input_ids          : (batch_size, sequence_length)
  //   Input 1 - segment_ids        : (batch_size, sequence_length)
  //   Input 2 - word_embedding     : (,hidden_size)
  //   Input 3 - position_embedding : (,hidden_size)
  //   Input 4 - segment_embedding  : (,hidden_size)
  //   Input 5 - gamma              : (hidden_size)
  //   Input 6 - beta               : (hidden_size)
  //   Input 7 - mask               : (batch_size, sequence_length)
  //   Output 0 - output            : (batch_size, sequence_length, hidden_size)
  //   Output 1 - mask_index        : (batch_size)

  OpTester tester("QEmbedLayerNormalization", 1, onnxruntime::kMSDomain);

  tester.AddInput<int32_t>("input_ids", input_ids_dims, input_ids_data);
  tester.AddInput<int32_t>("segment_ids", segment_ids_dims, segment_ids_data);

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

  std::vector<int32_t> input_ids_data = {
      1, 3};

  std::vector<int32_t> segment_ids_data = {
      0, 1};

  std::vector<float> output_data = {
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337};

  RunTest(input_ids_data,
          segment_ids_data,
          output_data,
          batch_size,
          sequence_length,
          hidden_size);
}  // namespace test

}  // namespace test
}  // namespace onnxruntime
