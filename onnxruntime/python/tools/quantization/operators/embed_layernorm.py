import onnx
from .base_operator import QuantOperatorBase
from onnx import onnx_pb as onnx_proto
'''
Quantize EmbedLayerNormalization
'''


class EmbedLayerNormalizationQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "EmbedLayerNormalization")

        import pdb; pdb.set_trace()
        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [2, 3, 4])

        import pdb; pdb.set_trace()

        #
        # NOTE: the method below will drop in the DequantizeLinear operators in the
        # graph if the node does not support fully quantized inputs.
        #
        # TODO - left off right here!
        #

        super().quantize()
        self.quantizer.new_nodes += nodes
