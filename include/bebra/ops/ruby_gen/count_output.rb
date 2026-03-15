COUNT_SHAPE_TEMPLATES = {
  elementwise: ->(op) {
    "return graph.getTensor(input).getShape();"
  },

  broadcast: ->(op) {
    <<-CPP
    auto shapeA = graph.getTensor(input_1).getShape();
    auto shapeB = graph.getTensor(input_2).getShape();
    return calculateBroadcastShape(shapeA, shapeB);
    CPP
  },

  reshape: ->(op) {
    if op == 'Flatten'
      "return calculateFlattenShape(graph.getTensor(input).getShape(), axis);"
    else
      "return graph.getTensor(shape).getShape();"
    end
  },

  maxpool: ->(op) {
  <<-CPP
    if (kernel_shape.size() != 2) {
        throw Core::BebraErr("only 2D MaxPool is supported for now!");
    }


    return {};
    CPP
  },

  reduce: ->(op) {
    <<-CPP
    auto inputShape = graph.getTensor(input).getShape();
    return calculateReduceShape(inputShape, axes, keepdims);
    CPP
  },


  spatial: ->(op) {
    <<-CPP
    auto inputShape = graph.getTensor(input).getShape();
    auto weightsShape = graph.getTensor(weight).getShape();

    if (inputShape.size() < 3) {
        throw Core::BebraErr("Input tensor rank must be >= 3 (N, C, H, ...)");
    }
    size_t spatial_dims = inputShape.size() - 2;

    int64_t outC = (strcmp(getOpType(), "Conv") == 0) ? weightsShape[0] : inputShape[1];

    std::vector<int64_t> outputShape;
    outputShape.push_back(inputShape[0]); // N
    outputShape.push_back(outC);          // C_out

    for (size_t i = 0; i < spatial_dims; ++i) {
        int64_t dim_in = inputShape[i + 2];
        int64_t kernel = kernel_shape[i];
        int64_t stride = (strides.size() > i) ? strides[i] : 1;

        int64_t p_start = 0;
        int64_t p_end = 0;

        if (pads.size() == spatial_dims * 2) {
            p_start = pads[i];
            p_end = pads[i + spatial_dims];
        } else if (pads.size() == spatial_dims) {
            p_start = p_end = pads[i];
        }


        int64_t effective_kernel = kernel;

        int64_t dim_out = (dim_in + p_start + p_end - effective_kernel) / stride + 1;
        outputShape.push_back(dim_out);
    }

    return outputShape;
    CPP
  },

  matmul: ->(op) {
    <<-CPP
    auto shapeA = graph.getTensor(input_a).getShape();
    auto shapeB = graph.getTensor(input_b).getShape();
    // [M, K] x [K, N] -> [M, N]
    return calculateMatMulShape(shapeA, shapeB);
    CPP
  },

  no_verify: ->(op) {
  <<-CPP
    return {};
  CPP
  }
}
