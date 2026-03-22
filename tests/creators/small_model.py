import onnx
from onnx import helper, TensorProto
import numpy as np

def create():
    input_a = helper.make_tensor_value_info('input_a', TensorProto.FLOAT, [1, 3, 32, 32])
    input_b = helper.make_tensor_value_info('input_b', TensorProto.FLOAT, [1, 3, 32, 32])

    const_value = np.array([2.0, 2.5, 2.5, 3.2], dtype=np.float32)
    const_tensor = helper.make_tensor(
        name='const_scale',
        data_type=TensorProto.FLOAT,
        dims=[1, 1, 1, 1],
        vals=const_value
    )

    node1 = helper.make_node('Add', inputs=['input_a', 'input_b'], outputs=['after_add'])
    node2 = helper.make_node('Relu', inputs=['after_add'], outputs=['after_relu'])
    node3 = helper.make_node('Mul', inputs=['after_relu', 'const_scale'], outputs=['output'])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])

    graph = helper.make_graph(
        nodes=[node1, node2, node3],
        name='small_3node_graph',
        inputs=[input_a, input_b],
        outputs=[output],
        initializer=[const_tensor]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])

    onnx.checker.check_model(model)
    onnx.save(model, './tiny_onnx/02_small_model.onnx')

    print("02_small_model.onnx created!")
