import onnx
from onnx import helper
from onnx import TensorProto


# node_def = helper.make_node("Add", ["X", "X"], ["Z"])
# graph = helper.make_graph(
#     [node_def],
#     "test",
#     [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
#     [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
#     doc_string=None,
#     value_info=None,
# )

node_def = helper.make_node("ReluGrad", ["X", "X"], ["Z"], domain="com.microsoft")
graph = helper.make_graph(
    [node_def],
    "test",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
    [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
    doc_string=None,
    value_info=None,
)

model = helper.make_model(graph)

opsets = [
    # ("", 12),
    ("com.microsoft.experimental", 1),
    ("ai.onnx.preview.training", 1),
    ("com.microsoft.nchwc", 1),
    ("com.microsoft.mlfeaturizers", 1),
    ("ai.onnx.ml", 2),
    ("com.microsoft", 1),
    ("ai.onnx.training", 1),
]
for opset in opsets:
    model.opset_import.add()
    model.opset_import[-1].domain = opset[0]
    model.opset_import[-1].version = opset[1]

onnx.save_model(model, "model.onnx")
