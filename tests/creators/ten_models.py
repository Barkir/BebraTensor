import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
from pathlib import Path

Path('./tiny_onnx').mkdir(exist_ok=True)


# ============================================================
# 1. Conv + BatchNorm + Relu (классическая CNN архитектура)
# ============================================================
def create_conv_bn_relu():
    """Классическая архитектура Conv -> BatchNorm -> Relu"""

    # Вход
    input_x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])

    # Conv веса [out_channels, in_channels, kH, kW]
    conv_w = np.random.randn(16, 3, 3, 3).astype(np.float32)
    conv_w_tensor = numpy_helper.from_array(conv_w, name='conv_weight')

    # BatchNorm параметры
    bn_scale = np.ones(16, dtype=np.float32)
    bn_bias = np.zeros(16, dtype=np.float32)
    bn_mean = np.zeros(16, dtype=np.float32)
    bn_var = np.ones(16, dtype=np.float32)

    conv_node = helper.make_node(
        'Conv', inputs=['input', 'conv_weight'],
        outputs=['conv_out'], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )

    bn_node = helper.make_node(
        'BatchNormalization',
        inputs=['conv_out', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
        outputs=['bn_out'], epsilon=1e-5
    )

    relu_node = helper.make_node('Relu', inputs=['bn_out'], outputs=['output'])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])

    graph = helper.make_graph(
        [conv_node, bn_node, relu_node],
        'conv_bn_relu',
        [input_x], [output],
        initializer=[conv_w_tensor]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.save(model, './tiny_onnx/01_conv_bn_relu.onnx')
    print("✓ 01_conv_bn_relu.onnx создан")


# ============================================================
# 2. GEMM (Fully Connected) + Softmax для классификации
# ============================================================
def create_gemm_classifier():
    """Полносвязная сеть для классификации"""

    batch_size = 4
    input_x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [batch_size, 128])

    # GEMM: Y = X @ W^T + B  (или X @ W + B если transB=1)
    fc1_w = np.random.randn(64, 128).astype(np.float32)
    fc1_b = np.random.randn(64).astype(np.float32)

    fc1_w_tensor = numpy_helper.from_array(fc1_w, name='fc1_weight')
    fc1_b_tensor = numpy_helper.from_array(fc1_b, name='fc1_bias')

    fc1_node = helper.make_node(
        'Gemm', inputs=['input', 'fc1_weight', 'fc1_bias'],
        outputs=['fc1_out'], alpha=1.0, beta=1.0, transA=0, transB=1
    )

    relu_node = helper.make_node('Relu', inputs=['fc1_out'], outputs=['relu_out'])

    # Выходной слой
    fc2_w = np.random.randn(10, 64).astype(np.float32)
    fc2_b = np.random.randn(10).astype(np.float32)

    fc2_w_tensor = numpy_helper.from_array(fc2_w, name='fc2_weight')
    fc2_b_tensor = numpy_helper.from_array(fc2_b, name='fc2_bias')

    fc2_node = helper.make_node(
        'Gemm', inputs=['relu_out', 'fc2_weight', 'fc2_bias'],
        outputs=['logits'], transB=1
    )

    softmax_node = helper.make_node(
        'Softmax', inputs=['logits'], outputs=['output'], axis=1
    )

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [batch_size, 10])

    graph = helper.make_graph(
        [fc1_node, relu_node, fc2_node, softmax_node],
        'gemm_classifier',
        [input_x], [output],
        initializer=[fc1_w_tensor, fc1_b_tensor, fc2_w_tensor, fc2_b_tensor]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.save(model, './tiny_onnx/02_gemm_classifier.onnx')
    print("✓ 02_gemm_classifier.onnx создан")


# ============================================================
# 3. Reshape + MatMul (Multi-layer Perceptron)
# ============================================================
def create_reshape_matmul():
    """MLP с явным reshape для совместимости с tosa.matmul (3D тензоры)"""
    import numpy as np
    from onnx import helper, TensorProto, numpy_helper, save

    # Вход: [1, 1, 28, 28]
    input_x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 28, 28])

    # === Flatten: [1, 1, 28, 28] -> [1, 784] ===
    flatten_shape = np.ascontiguousarray([1, 784], dtype=np.int64)
    flatten_shape_tensor = numpy_helper.from_array(flatten_shape, name='flatten_shape')
    flatten_node = helper.make_node(
        'Reshape', inputs=['input', 'flatten_shape'], outputs=['flattened']
    )

    # === Слой 1: Подготовка к 3D MatMul ===
    # Веса ONNX: [out, in] = [256, 784]. Для TOSA нужно [B, K, N] = [1, 784, 256]
    w1 = np.ascontiguousarray(np.random.randn(256, 784).astype(np.float32))
    w1_t = w1.T  # Транспонируем: [784, 256]
    w1_3d = w1_t.reshape(1, 784, 256)  # Добавляем батч: [1, 784, 256]

    # Bias должен соответствовать выходу MatMul: [1, 1, 256]
    b1 = np.ascontiguousarray(np.random.randn(256).astype(np.float32))
    b1_3d = b1.reshape(1, 1, 256)

    w1_tensor = numpy_helper.from_array(w1_3d, name='w1')
    b1_tensor = numpy_helper.from_array(b1_3d, name='b1')

    # Reshape перед умножением: [1, 784] -> [1, 1, 784]
    input_shape_1 = np.ascontiguousarray([1, 1, 784], dtype=np.int64)
    input_shape_1_tensor = numpy_helper.from_array(input_shape_1, name='input_shape_1')
    reshape_in1_node = helper.make_node(
        'Reshape', inputs=['flattened', 'input_shape_1'], outputs=['input_3d']
    )

    # MatMul: [1, 1, 784] @ [1, 784, 256] -> [1, 1, 256]
    mm1_node = helper.make_node(
        'MatMul', inputs=['input_3d', 'w1'], outputs=['mm1_out']
    )

    add1_node = helper.make_node(
        'Add', inputs=['mm1_out', 'b1'], outputs=['dense1_out']
    )
    act1_node = helper.make_node('Relu', inputs=['dense1_out'], outputs=['act1_out'])

    # === Слой 2: Подготовка к 3D MatMul ===
    # Веса ONNX: [10, 256]. Для TOSA нужно [1, 256, 10]
    w2 = np.ascontiguousarray(np.random.randn(10, 256).astype(np.float32))
    w2_t = w2.T  # [256, 10]
    w2_3d = w2_t.reshape(1, 256, 10)  # [1, 256, 10]

    # Bias: [1, 1, 10]
    b2 = np.ascontiguousarray(np.random.randn(10).astype(np.float32))
    b2_3d = b2.reshape(1, 1, 10)

    w2_tensor = numpy_helper.from_array(w2_3d, name='w2')
    b2_tensor = numpy_helper.from_array(b2_3d, name='b2')

    # Reshape перед умножением: [1, 256] -> [1, 1, 256]
    # (act1_out имеет форму [1, 1, 256] после Add, но для явности можно добавить reshape)
    # Если Add сохранил 3D, этот ресайп может быть опущен, но для надежности оставим
    input_shape_2 = np.ascontiguousarray([1, 1, 256], dtype=np.int64)
    input_shape_2_tensor = numpy_helper.from_array(input_shape_2, name='input_shape_2')
    reshape_in2_node = helper.make_node(
        'Reshape', inputs=['act1_out', 'input_shape_2'], outputs=['input_3d_2']
    )

    # MatMul: [1, 1, 256] @ [1, 256, 10] -> [1, 1, 10]
    mm2_node = helper.make_node('MatMul', inputs=['input_3d_2', 'w2'], outputs=['output_raw'])
    add2_node = helper.make_node('Add', inputs=['output_raw', 'b2'], outputs=['output_3d'])

    # Финальный Reshape: [1, 1, 10] -> [1, 10] (классический выход)
    final_shape = np.ascontiguousarray([1, 10], dtype=np.int64)
    final_shape_tensor = numpy_helper.from_array(final_shape, name='final_shape')
    final_reshape_node = helper.make_node(
        'Reshape', inputs=['output_3d', 'final_shape'], outputs=['output']
    )

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])

    graph = helper.make_graph(
        [flatten_node, reshape_in1_node, mm1_node, add1_node, act1_node,
         reshape_in2_node, mm2_node, add2_node, final_reshape_node],
        'reshape_matmul_mlp_3d',
        [input_x], [output],
        initializer=[flatten_shape_tensor, input_shape_1_tensor, w1_tensor, b1_tensor,
                     input_shape_2_tensor, w2_tensor, b2_tensor, final_shape_tensor]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])

    # Валидация
    from onnx.checker import check_model
    check_model(model)

    save(model, './tiny_onnx/03_reshape_matmul_3d.onnx')
    print("✓ 03_reshape_matmul_3d.onnx создан")


# ============================================================
# 4. MaxPool + AveragePool + GlobalPooling
# ============================================================
def create_pooling_stages():
    """Различные типы pooling операций"""

    input_x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 64, 24, 24])

    # MaxPool
    maxpool_node = helper.make_node(
        'MaxPool',
        inputs=['input'],
        outputs=['maxpool_out'],
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0]
    )

    # AveragePool
    avgpool_node = helper.make_node(
        'AveragePool',
        inputs=['maxpool_out'],
        outputs=['avgpool_out'],
        kernel_shape=[2, 2],
        strides=[2, 2]
    )

    # Global Average Pool - превращаем [1, 64, 6, 6] -> [1, 64]
    # Для этого нужен ReduceMean с keepdims=0
    gap_node = helper.make_node(
        'GlobalAveragePool',
        inputs=['avgpool_out'],
        outputs=['gap_out']
    )

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 64])

    graph = helper.make_graph(
        [maxpool_node, avgpool_node, gap_node],
        'pooling_stages',
        [input_x], [output]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.save(model, './tiny_onnx/04_pooling_stages.onnx')
    print("✓ 04_pooling_stages.onnx создан")


# ============================================================
# 5. Concat + Split (разделение и объединение каналов)
# ============================================================
def create_concat_split():
    """Split по каналам, обработка и Concat обратно"""

    batch, channels, height, width = 1, 8, 16, 16
    half = channels // 2

    input_x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [batch, channels, height, width])

    # Split на 2 части по каналам
    split_node = helper.make_node(
        'Split',
        inputs=['input'],
        outputs=[f'split_{i}' for i in range(2)],
        axis=1,
        split=[half, half]
    )

    # Разные операции для каждой части
    conv1_w = np.random.randn(4, 4, 1, 1).astype(np.float32)
    conv1_w_tensor = numpy_helper.from_array(conv1_w, name='conv1_w')

    conv1_node = helper.make_node(
        'Conv',
        inputs=['split_0', 'conv1_w'],
        outputs=['conv1_out'],
        kernel_shape=[1, 1]
    )

    relu1_node = helper.make_node('Relu', inputs=['conv1_out'], outputs=['path1_out'])

    # Вторая ветка - pooling
    pool2_node = helper.make_node(
        'MaxPool',
        inputs=['split_1'],
        outputs=['path2_out'],
        kernel_shape=[2, 2],
        strides=[2, 2]
    )

    # Upsample для выравнивания размеров
    # Или используем GlobalAveragePool
    gap_node = helper.make_node(
        'GlobalAveragePool',
        inputs=['path2_out'],
        outputs=['path2_gap_out']
    )

    # Reshape для восстановления размерности
    reshape_shape = np.array([1, 4, 1, 1], dtype=np.int64)
    reshape_shape_tensor = numpy_helper.from_array(reshape_shape, name='reshape_shape')

    reshape_node = helper.make_node(
        'Reshape',
        inputs=['path2_gap_out', 'reshape_shape'],
        outputs=['path2_final']
    )

    # Tile для расширения до нужного размера
    tile_node = helper.make_node(
        'Tile',
        inputs=['path2_final'],
        outputs=['path2_tiled'],
        repeats=[1, 1, 16, 16]
    )

    # Concat обратно
    concat_node = helper.make_node(
        'Concat',
        inputs=['path1_out', 'path2_tiled'],
        outputs=['output'],
        axis=1
    )

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 8, 16, 16])

    graph = helper.make_graph(
        [split_node, conv1_node, relu1_node, pool2_node, gap_node, reshape_node, tile_node, concat_node],
        'concat_split',
        [input_x], [output],
        initializer=[conv1_w_tensor, reshape_shape_tensor]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.save(model, './tiny_onnx/05_concat_split.onnx')
    print("✓ 05_concat_split.onnx создан")


# ============================================================
# 6. Transpose + Gather для attention-like операции
# ============================================================
def create_transpose_gather():
    """Модель с транспонированием и выборкой элементов"""

    batch, seq_len, hidden = 2, 8, 16

    input_x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [batch, seq_len, hidden])

    # Транспонируем: [B, S, H] -> [B, H, S]
    transpose_node = helper.make_node(
        'Transpose',
        inputs=['input'],
        outputs=['transposed'],
        perm=[0, 2, 1]
    )

    # MatMul для attention-like весов
    qkv_w = np.random.randn(3 * hidden, hidden).astype(np.float32)
    qkv_w_tensor = numpy_helper.from_array(qkv_w, name='qkv_weight')

    # Reshape для QKV projection
    reshape_shape = np.array([batch, seq_len, 3, hidden], dtype=np.int64)
    reshape_shape_tensor = numpy_helper.from_array(reshape_shape, name='reshape_shape')

    mm_node = helper.make_node(
        'MatMul',
        inputs=['input', 'qkv_weight'],
        outputs=['mm_out']
    )

    reshape_node = helper.make_node(
        'Reshape',
        inputs=['mm_out', 'reshape_shape'],
        outputs=['reshaped']
    )

    # Transpose для получения [B, 3, H, S]
    transpose2_node = helper.make_node(
        'Transpose',
        inputs=['reshaped'],
        outputs=['qkv_transposed'],
        perm=[0, 2, 3, 1]
    )

    # Squeeze для извлечения Q, K, V
    # Slice или Split для извлечения
    slice_starts = np.array([0], dtype=np.int64)
    slice_ends = np.array([hidden], dtype=np.int64)
    slice_axes = np.array([2], dtype=np.int64)

    q_node = helper.make_node(
        'Slice',
        inputs=['qkv_transposed', 'slice_starts', 'slice_ends', 'slice_axes'],
        outputs=['q']
    )

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [batch, seq_len, hidden])

    graph = helper.make_graph(
        [transpose_node, mm_node, reshape_node, transpose2_node, q_node],
        'transpose_gather',
        [input_x], [output],
        initializer=[qkv_w_tensor, reshape_shape_tensor, slice_starts, slice_ends, slice_axes]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.save(model, './tiny_onnx/06_transpose_gather.onnx')
    print("✓ 06_transpose_gather.onnx создан")


# ============================================================
# 7. DepthwiseSeparable Convolution (MobileNet-style)
# ============================================================
def create_depthwise_separable_conv():
    """Depthwise + Pointwise convolution (MobileNet блок)"""

    input_x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])

    # Depthwise Conv: groups=in_channels
    dw_w = np.random.randn(3, 1, 3, 3).astype(np.float32)
    dw_w_tensor = numpy_helper.from_array(dw_w, name='depthwise_weight')

    dw_conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'depthwise_weight'],
        outputs=['dw_out'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        group=3  # depthwise
    )

    dw_bn_scale = np.ones(3, dtype=np.float32)
    dw_bn_bias = np.zeros(3, dtype=np.float32)
    dw_bn_mean = np.zeros(3, dtype=np.float32)
    dw_bn_var = np.ones(3, dtype=np.float32)

    dw_bn_node = helper.make_node(
        'BatchNormalization',
        inputs=['dw_out', 'dw_bn_scale', 'dw_bn_bias', 'dw_bn_mean', 'dw_bn_var'],
        outputs=['dw_bn_out']
    )

    dw_relu_node = helper.make_node('Relu', inputs=['dw_bn_out'], outputs=['dw_relu_out'])

    # Pointwise Conv: 1x1 Conv без groups
    pw_w = np.random.randn(32, 3, 1, 1).astype(np.float32)
    pw_w_tensor = numpy_helper.from_array(pw_w, name='pointwise_weight')

    pw_conv_node = helper.make_node(
        'Conv',
        inputs=['dw_relu_out', 'pointwise_weight'],
        outputs=['pw_out'],
        kernel_shape=[1, 1]
    )

    pw_bn_scale = np.ones(32, dtype=np.float32)
    pw_bn_bias = np.zeros(32, dtype=np.float32)
    pw_bn_mean = np.zeros(32, dtype=np.float32)
    pw_bn_var = np.ones(32, dtype=np.float32)

    pw_bn_node = helper.make_node(
        'BatchNormalization',
        inputs=['pw_out', 'pw_bn_scale', 'pw_bn_bias', 'pw_bn_mean', 'pw_bn_var'],
        outputs=['output']
    )

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 32, 224, 224])

    graph = helper.make_graph(
        [dw_conv_node, dw_bn_node, dw_relu_node, pw_conv_node, pw_bn_node],
        'depthwise_separable_conv',
        [input_x], [output],
        initializer=[dw_w_tensor, pw_w_tensor]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.save(model, './tiny_onnx/07_depthwise_separable.onnx')
    print("✓ 07_depthwise_separable.onnx создан")


# ============================================================
# 8. Residual/Skip Connection (ResNet-style block)
# ============================================================
def create_residual_block():
    """ResNet-подобный блок с skip connection"""

    input_x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 64, 56, 56])

    # Main path
    conv1_w = np.random.randn(64, 64, 3, 3).astype(np.float32)
    conv1_w_tensor = numpy_helper.from_array(conv1_w, name='conv1_weight')

    conv1_node = helper.make_node(
        'Conv',
        inputs=['input', 'conv1_weight'],
        outputs=['conv1_out'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1]
    )

    bn1_node = helper.make_node(
        'BatchNormalization',
        inputs=['conv1_out', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
        outputs=['bn1_out']
    )
    relu1_node = helper.make_node('Relu', inputs=['bn1_out'], outputs=['relu1_out'])

    conv2_w = np.random.randn(64, 64, 3, 3).astype(np.float32)
    conv2_w_tensor = numpy_helper.from_array(conv2_w, name='conv2_weight')

    conv2_node = helper.make_node(
        'Conv',
        inputs=['relu1_out', 'conv2_weight'],
        outputs=['conv2_out'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1]
    )

    bn2_node = helper.make_node(
        'BatchNormalization',
        inputs=['conv2_out', 'bn2_scale', 'bn2_bias', 'bn2_mean', 'bn2_var'],
        outputs=['bn2_out']
    )

    # Skip connection (identity)
    add_node = helper.make_node(
        'Add',
        inputs=['bn2_out', 'input'],
        outputs=['add_out']
    )

    relu2_node = helper.make_node('Relu', inputs=['add_out'], outputs=['output'])

    # BN параметры
    bn_scale = np.ones(64, dtype=np.float32)
    bn_bias = np.zeros(64, dtype=np.float32)
    bn_mean = np.zeros(64, dtype=np.float32)
    bn_var = np.ones(64, dtype=np.float32)

    bn2_scale = np.ones(64, dtype=np.float32)
    bn2_bias = np.zeros(64, dtype=np.float32)
    bn2_mean = np.zeros(64, dtype=np.float32)
    bn2_var = np.ones(64, dtype=np.float32)

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 64, 56, 56])

    graph = helper.make_graph(
        [conv1_node, bn1_node, relu1_node, conv2_node, bn2_node, add_node, relu2_node],
        'residual_block',
        [input_x], [output],
        initializer=[conv1_w_tensor, conv2_w_tensor]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.save(model, './tiny_onnx/08_residual_block.onnx')
    print("✓ 08_residual_block.onnx создан")


# ============================================================
# 9. Pad + PadConv + Unpad (padding операции)
# ============================================================
def create_pad_convolution():
    """Модель с различными pad операциями"""

    input_x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 28, 28])

    # Pad для расширения изображения
    # pads = [batch, channel, top, bottom, left, right] для NCHW
    pads = np.array([0, 0, 2, 2, 2, 2], dtype=np.int64)
    pads_tensor = numpy_helper.from_array(pads, name='pads')

    pad_node = helper.make_node(
        'Pad',
        inputs=['input', 'pads'],
        outputs=['padded'],
        mode='constant',
        constant_value=0.0
    )

    # Conv после padding
    conv_w = np.random.randn(16, 3, 5, 5).astype(np.float32)
    conv_w_tensor = numpy_helper.from_array(conv_w, name='conv_weight')

    conv_node = helper.make_node(
        'Conv',
        inputs=['padded', 'conv_weight'],
        outputs=['conv_out'],
        kernel_shape=[5, 5]
    )

    relu_node = helper.make_node('Relu', inputs=['conv_out'], outputs=['output'])

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 28, 28])

    graph = helper.make_graph(
        [pad_node, conv_node, relu_node],
        'pad_convolution',
        [input_x], [output],
        initializer=[pads_tensor, conv_w_tensor]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.save(model, './tiny_onnx/09_pad_convolution.onnx')
    print("✓ 09_pad_convolution.onnx создан")


# ============================================================
# 10. сложная модель: Element-wise + Reduce + Expand
# ============================================================
def create_complex_elementwise():
    """Сложная модель с множеством element-wise операций"""

    input_a = helper.make_tensor_value_info('input_a', TensorProto.FLOAT, [1, 64, 32, 32])
    input_b = helper.make_tensor_value_info('input_b', TensorProto.FLOAT, [1, 64, 32, 32])

    # Element-wise операции
    add_node = helper.make_node('Add', inputs=['input_a', 'input_b'], outputs=['added'])

    mul_node = helper.make_node('Mul', inputs=['added', 'input_a'], outputs=['multiplied'])

    sub_node = helper.make_node('Sub', inputs=['input_b', 'input_a'], outputs=['subtracted'])

    div_node = helper.make_node('Div', inputs=['added', 'subtracted'], outputs=['divided'])

    # Reduce операции
    reduce_mean_node = helper.make_node(
        'ReduceMean',
        inputs=['divided'],
        outputs=['reduced'],
        axes=[2, 3],
        keepdims=1
    )

    # Squeeze для удаления размерности 1
    squeeze_axes = np.array([2, 3], dtype=np.int64)
    squeeze_axes_tensor = numpy_helper.from_array(squeeze_axes, name='squeeze_axes')

    squeeze_node = helper.make_node(
        'Squeeze',
        inputs=['reduced', 'squeeze_axes'],
        outputs=['squeezed']
    )

    # Expand обратно
    expand_shape = np.array([1, 64, 1, 1], dtype=np.int64)
    expand_shape_tensor = numpy_helper.from_array(expand_shape, name='expand_shape')

    expand_node = helper.make_node(
        'Reshape',
        inputs=['squeezed', 'expand_shape'],
        outputs=['expanded']
    )

    # Tile для расширения
    repeats = np.array([1, 1, 32, 32], dtype=np.int64)
    repeats_tensor = numpy_helper.from_array(repeats, name='repeats')

    tile_node = helper.make_node(
        'Tile',
        inputs=['expanded', 'repeats'],
        outputs=['tiled']
    )

    # Финальная операция с tiled tensor
    final_mul_node = helper.make_node(
        'Mul',
        inputs=['divided', 'tiled'],
        outputs=['output']
    )

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 64, 32, 32])

    graph = helper.make_graph(
        [add_node, mul_node, sub_node, div_node, reduce_mean_node,
         squeeze_node, expand_node, tile_node, final_mul_node],
        'complex_elementwise',
        [input_a, input_b], [output],
        initializer=[squeeze_axes_tensor, expand_shape_tensor, repeats_tensor]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.save(model, './tiny_onnx/10_complex_elementwise.onnx')
    print("✓ 10_complex_elementwise.onnx создан")


# ============================================================
# Генерация всех моделей
# ============================================================
def generate_all_models():
    """Генерирует все 10 тестовых моделей"""

    print("=" * 60)
    print("Генерация ONNX моделей для тестирования AI-компилятора")
    print("=" * 60)

    models = [
        create_conv_bn_relu,
        create_gemm_classifier,
        create_reshape_matmul,
        create_pooling_stages,
        create_concat_split,
        create_transpose_gather,
        create_depthwise_separable_conv,
        create_residual_block,
        create_pad_convolution,
        create_complex_elementwise,
    ]

    for i, model_fn in enumerate(models, 1):
        print(f"\n[{i}/10] Генерация {model_fn.__name__}...")
        try:
            model_fn()
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")

    print("\n" + "=" * 60)
    print("Все модели успешно сгенерированы в ./tiny_onnx/")
    print("=" * 60)

    # Список всех созданных моделей
    print("\nСозданные модели:")
    import os
    for f in sorted(os.listdir('./tiny_onnx')):
        if f.endswith('.onnx'):
            path = os.path.join('./tiny_onnx', f)
            size = os.path.getsize(path) / 1024
            print(f"  • {f} ({size:.1f} KB)")


if __name__ == '__main__':
    generate_all_models()
