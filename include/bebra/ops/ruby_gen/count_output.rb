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
  const auto& inputShape = graph.getTensor(input).getShape();
    if (inputShape.size() < 3) {
        throw Core::BebraErr("MaxPool: input tensor must have at least 3 dimensions [N, C, D1...]");
    }

    std::vector<int64_t> outputShape;
    outputShape.push_back(inputShape[0]); // N
    outputShape.push_back(inputShape[1]); // C

    size_t spatial_dims = kernel_shape.size();
    for (size_t i = 0; i < spatial_dims; ++i) {
        int64_t dim_in = inputShape[i + 2];
        int64_t d_kernel = dilations[i] * (kernel_shape[i] - 1) + 1;
        int64_t dim_out = 0  ;

        if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
            dim_out = (dim_in + strides[i] - 1) / strides[i];
        } else {
            // "NOTSET" or "VALID"
            int64_t p_total = (auto_pad != "NOTSET") ? 0 : (pads[i] + pads[i + spatial_dims]);
            double val = static_cast<double>(dim_in + p_total - d_kernel) / static_cast<double>(strides[i]) + 1.0;

            if (ceil_mode) {
                dim_out = static_cast<int64_t>(std::ceil(val));
            } else {
                dim_out = static_cast<int64_t>(std::floor(val));
            }
        }
        outputShape.push_back(dim_out);
    }
    return outputShape;
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
    auto inputShape = graph.getTensor(input).getShape(); // [N, C, H, W]
    auto weightsShape = graph.getTensor(weight).getShape();
    int64_t outH = (inputShape[2] + pads[0] + pads[2] - kernel_shape[0]) / strides[0] + 1;
    int64_t outW = (inputShape[3] + pads[1] + pads[3] - kernel_shape[1]) / strides[1] + 1;

    auto op_type = getOpType();

    return { inputShape[0], (!strcmp(op_type, "Conv") ? weightsShape[0] : inputShape[1]), outH, outW };
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
