import onnx
from onnx import helper
from onnx import TensorProto


def add_opset(model):
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


def make_model(nodes, inputs, outputs):
    graph = helper.make_graph(
        nodes,
        "test",
        inputs,
        outputs,
        doc_string=None,
        value_info=None,
    )
    model = helper.make_model(graph)
    add_opset(model)
    return model


def make_fwd_test():
    node_def = helper.make_node("Add", ["X", "X"], ["Z"])
    model = make_model(
        [node_def],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
    )
    onnx.save_model(model, "model.onnx")


def make_bwd_test():
    node_def = helper.make_node("ReluGrad", ["X", "X"], ["Z"], domain="com.microsoft")
    model = make_model(
        [node_def],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
    )
    onnx.save_model(model, "model.onnx")


def make_send_recv_test():
    node_def = helper.make_node(
        "Send",
        ["input_signal_token", "dst_rank_token", "X"],
        ["output_signal"],
        domain="com.microsoft",
        element_types=[1],  # Type of each sent tensor
        tag=0,
    )
    inputs = [
        helper.make_tensor_value_info(
            "input_signal_token", TensorProto.BOOL, shape=None
        ),
        helper.make_tensor_value_info("dst_rank_token", TensorProto.INT64, shape=None),
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2]),
    ]
    model = make_model(
        [node_def],
        inputs,
        [helper.make_tensor_value_info("output_signal", TensorProto.BOOL, shape=None)],
    )
    onnx.save_model(model, "model-0.onnx")

    node_def = helper.make_node(
        "Recv",
        ["input_signal_token", "src_rank_token"],
        ["output_signal", "X"],
        domain="com.microsoft",
        element_types=[1],  # Type of each sent tensor
        tag=0,
    )
    inputs = [
        helper.make_tensor_value_info(
            "input_signal_token", TensorProto.BOOL, shape=None
        ),
        helper.make_tensor_value_info("src_rank_token", TensorProto.INT64, shape=None),
    ]
    outputs = [
        helper.make_tensor_value_info("output_signal", TensorProto.BOOL, shape=None),
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2]),
    ]
    model = make_model(
        [node_def],
        inputs,
        outputs,
    )
    onnx.save_model(model, "model-1.onnx")


if __name__ == "__main__":
    # make_fwd_test()
    # make_bwd_test()
    make_send_recv_test()
