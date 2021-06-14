from numpy import kaiser
import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import attribute_to_kwarg, ms_domain
from onnx import onnx_pb as onnx_proto
'''
Quantize EmbedLayerNormalization
TODO(kreeger): Add more documentation here.
'''


class EmbedLayerNormalizationQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "EmbedLayerNormalization")

        # TODO(kreeger): what is the |reduce_range| flag here?
        '''
        Pre-quantized inputs:
        [0] input_ids (int32)
        [1] segmentd_ids (int32)
        [2] word_embedding (float32)
        [3] position_embedding (float32)
        [4] segment_embedding (float32)
        [5] layer_norm_weight (float32) 
        [6] layer_norm_bias (float32)
        [7] mask (int32) (optional)
        '''
        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [2, 3, 4, 5, 6])

        # TODO(kreeger): Unit test for this (seems empty str in the unit test right now):
        qembed_layer_norm_name = "" if node.name == "" else node.name + "_quant"

        # TODO(kreeger): Check inputs for gamma/beta/mask with len(node.input) > N
        ''' 
        Tensors List:
        [0] input_ids
        [1] segmend_ids
        [2] word_embedding_quant
        [3] word_embedding_scale
        [4] word_embedding_zero_point
        [5] position_embedding_quant
        [6] position_embedding_scale
        [7] position_embedding_zp
        [8] segment_embedding_quant
        [9] segment_embedding_scale
        [10] segment_embedding_zp
        [11] layer_norm_weights_quant
        [11] layer_norm_weights_scale
        [12] layer_norm_weights_zero_point
        [13] layer_norm_bias_quant
        [14] layer_norm_bias_scale
        [15] layer_norm_bias_zero_point

        [16] mask (quant/support?)
        '''
        inputs = []

        # 'input_ids'
        inputs.extend([node.input[0]])

        # 'segment_ids'
        inputs.extend([node.input[1]])

        # 'word_embedding_quant', 'word_embedding_scale', 'word_embedding_zero_point'
        inputs.extend([quantized_input_names[0]])
        inputs.extend([scale_names[0]])
        inputs.extend([zero_point_names[0]])

        # 'position_embedding_quant', 'position_embedding_scale', 'position_embedding_zero_point'
        inputs.extend([quantized_input_names[1]])
        inputs.extend([scale_names[1]])
        inputs.extend([zero_point_names[1]])

        # 'segment_embedding_quant', 'segment_embedding_scale', 'segment_embedding_zero_point'
        inputs.extend([quantized_input_names[2]])
        inputs.extend([scale_names[2]])
        inputs.extend([zero_point_names[2]])

        # 'segment_embedding_quant', 'segment_embedding_scale', 'segment_embedding_zero_point'
        inputs.extend([quantized_input_names[2]])
        inputs.extend([scale_names[2]])
        inputs.extend([zero_point_names[2]])

        # 'layer_norm_weights_quant', 'layer_norm_weights_scale', 'layer_norm_weights_zero_point'
        inputs.extend([quantized_input_names[3]])
        inputs.extend([scale_names[3]])
        inputs.extend([zero_point_names[3]])

        # 'layer_norm_bias_quant', 'layer_norm_bias_scale', 'layer_norm_bias_zero_point'
        inputs.extend([quantized_input_names[4]])
        inputs.extend([scale_names[4]])
        inputs.extend([zero_point_names[4]])

        # mask
        if len(node.input) > 7:
            inputs.extend([node.input[7]])

        print(inputs)

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        # TODO(kreeger): Update this doc here.
        # NOTE: since ORT outputs zp & scale as tensors this graph needs some additional output.
        # The consumer of layernorm_out* tends to be a QAttention layer that already takes quantized
        # inputs with the additional tensors through a DynamicQuantizeLinear Op.
        qembed_layer_norm_node = onnx.helper.make_node("QEmbedLayerNormalization", inputs, node.output,
                                                       qembed_layer_norm_name, **kwargs)
        nodes.append(qembed_layer_norm_node)

        self.quantizer.new_nodes += nodes

        import pdb
        pdb.set_trace()
