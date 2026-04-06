import onnx
from onnx import helper, TensorProto
import numpy as np

def create_linear_model(output_path="simple_linear.onnx"):
    # 1. Define Input and Output
    # Shape: [Batch, Features] -> [1, 4]
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

    # 2. Define Weights and Biases (Initializers)
    # Layer 1: 4 inputs -> 8 hidden units
    w1_data = np.random.randn(4, 8).astype(np.float32)
    b1_data = np.random.randn(1, 8).astype(np.float32)
    # Layer 2: 8 hidden units -> 2 outputs
    w2_data = np.random.randn(8, 2).astype(np.float32)
    b2_data = np.random.randn(1, 2).astype(np.float32)

    w1 = helper.make_tensor('W1', TensorProto.FLOAT, [4, 8], w1_data.flatten())
    b1 = helper.make_tensor('B1', TensorProto.FLOAT, [1, 8], b1_data.flatten())
    w2 = helper.make_tensor('W2', TensorProto.FLOAT, [8, 2], w2_data.flatten())
    b2 = helper.make_tensor('B2', TensorProto.FLOAT, [1, 2], b2_data.flatten())

    # 3. Create Nodes (MatMul + Add instead of Gemm to mirror TOSA logic)
    node1 = helper.make_node('MatMul', ['input', 'W1'], ['mm1'])
    node2 = helper.make_node('Add', ['mm1', 'B1'], ['add1'])
    node3 = helper.make_node('Relu', ['add1'], ['relu1'])
    node4 = helper.make_node('MatMul', ['relu1', 'W2'], ['mm2'])
    node5 = helper.make_node('Add', ['mm2', 'B2'], ['output'])

    # 4. Assemble the Graph
    graph = helper.make_graph(
        [node1, node2, node3, node4, node5],
        'SimpleLinearModel',
        [input_info],
        [output_info],
        [w1, b1, w2, b2]
    )

    # 5. Create and save the Model
    model = helper.make_model(graph, producer_name='onnx-simple-creator')
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    create_linear_model()
