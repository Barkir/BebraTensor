import onnx
from onnx import helper, TensorProto
import numpy as np
import os

def create_arithmetic_model():
    os.makedirs('./test_models', exist_ok=True)

    input_a = helper.make_tensor_value_info('input_a', TensorProto.FLOAT, [1, 3, 32, 32])
    input_b = helper.make_tensor_value_info('input_b', TensorProto.FLOAT, [1, 3, 32, 32])

    const_value = np.array([2.0], dtype=np.float32)
    const_tensor = helper.make_tensor(
        name='scale',
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=const_value
    )

    node_add = helper.make_node('Add', inputs=['input_a', 'input_b'], outputs=['sum'])
    node_relu = helper.make_node('Relu', inputs=['sum'], outputs=['relu_out'])
    node_mul = helper.make_node('Mul', inputs=['relu_out', 'scale'], outputs=['output'])

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])

    graph = helper.make_graph(
        nodes=[node_add, node_relu, node_mul],
        name='ArithmeticGraph',
        inputs=[input_a, input_b],
        outputs=[output],
        initializer=[const_tensor]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.checker.check_model(model)
    onnx.save(model, './tiny_onnx/01_arithmetic.onnx')
    print("Created: 01_arithmetic.onnx")

if __name__ == "__main__":
    create_arithmetic_model()
