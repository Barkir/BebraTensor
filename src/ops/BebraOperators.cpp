#include "bebra/ops/BebraOperators.hpp"
#include "bebra/core/BebraGraph.hpp"
#include "bebra/ops/CountShapeHelpers.hpp"
namespace Bebra::Ops {
bool OpVoid::verify(const Core::BebraGraph& graph) const {
    return true;
}
bool OpConv::verify(const Core::BebraGraph& graph) const {

        const auto& tensor_in = graph.getTensor(input);

        if (!tensor_in.hasShape()) {
            return true;
        }

        const auto& shape = tensor_in.shape_;

        if (shape.size() < 3) {
            std::cerr << "Spatial op requires at least 3D input, got "
                      << shape.size() << "D" << std::endl;
            return false;
        }


        size_t spatial_dims = shape.size() - 2;
        if (kernel_shape.size() != spatial_dims) {
            std::cerr << "Kernel shape rank " << kernel_shape.size()
                      << " doesn't match input spatial dims " << spatial_dims << std::endl;
            return false;
        }

        if (!strides.empty() && strides.size() != spatial_dims) {
            std::cerr << "Strides rank mismatch" << std::endl;
            return false;
        }

        if (!dilations.empty() && dilations.size() != spatial_dims) {
            std::cerr << "Dilations rank mismatch" << std::endl;
            return false;
        }

        if (!pads.empty() && pads.size() != spatial_dims && pads.size() != 2 * spatial_dims) {
            std::cerr << "Pads rank mismatch" << std::endl;
            return false;
        }


        for (int64_t k : kernel_shape) {
            if (k <= 0) {
                std::cerr << "Kernel shape must be positive, got " << k << std::endl;
                return false;
            }
        }


        for (int64_t s : strides) {
            if (s <= 0) {
                std::cerr << "Strides must be positive, got " << s << std::endl;
                return false;
            }
        }

        return true;
}
bool OpGemm::verify(const Core::BebraGraph& graph) const {
    return true;
}
bool OpAdd::verify(const Core::BebraGraph& graph) const {

        const auto& tensor_1 = graph.getTensor(input_1);
        const auto& tensor_2 = graph.getTensor(input_2);

        if (!tensor_1.hasShape() || !tensor_2.hasShape()) {
            return true;
        }

        const auto& shape_1 = tensor_1.shape_;
        const auto& shape_2 = tensor_2.shape_;


        size_t rank_1 = shape_1.size();
        size_t rank_2 = shape_2.size();
        size_t max_rank = std::max(rank_1, rank_2);

        for (size_t i = 0; i < max_rank; ++i) {
            int64_t dim_1 = (i < rank_1) ? shape_1[rank_1 - 1 - i] : 1;
            int64_t dim_2 = (i < rank_2) ? shape_2[rank_2 - 1 - i] : 1;

            if (dim_1 != dim_2 && dim_1 != 1 && dim_2 != 1) {
                std::cerr << "Broadcast incompatible at dim " << i
                          << ": " << dim_1 << " vs " << dim_2 << std::endl;
                return false;
            }
        }

        return true;
}
bool OpRelu::verify(const Core::BebraGraph& graph) const {

        const auto& tensor_in = graph.getTensor(input);

        if (!tensor_in.hasShape()) {
            return true;
        }

        for (int64_t dim : tensor_in.shape_) {
            if (dim <= 0) {
                std::cerr << "Invalid dimension size " << dim << std::endl;
                return false;
            }
        }

        return true;
}
bool OpMul::verify(const Core::BebraGraph& graph) const {

        const auto& tensor_1 = graph.getTensor(input_1);
        const auto& tensor_2 = graph.getTensor(input_2);

        if (!tensor_1.hasShape() || !tensor_2.hasShape()) {
            return true;
        }

        const auto& shape_1 = tensor_1.shape_;
        const auto& shape_2 = tensor_2.shape_;


        size_t rank_1 = shape_1.size();
        size_t rank_2 = shape_2.size();
        size_t max_rank = std::max(rank_1, rank_2);

        for (size_t i = 0; i < max_rank; ++i) {
            int64_t dim_1 = (i < rank_1) ? shape_1[rank_1 - 1 - i] : 1;
            int64_t dim_2 = (i < rank_2) ? shape_2[rank_2 - 1 - i] : 1;

            if (dim_1 != dim_2 && dim_1 != 1 && dim_2 != 1) {
                std::cerr << "Broadcast incompatible at dim " << i
                          << ": " << dim_1 << " vs " << dim_2 << std::endl;
                return false;
            }
        }

        return true;
}
bool OpMatMul::verify(const Core::BebraGraph& graph) const {

 const auto& tensor_a = graph.getTensor(input_a);
    const auto& tensor_b = graph.getTensor(input_b);

    if (!tensor_a.hasShape() || !tensor_b.hasShape()) {
        return true;
    }

    const auto& shape_a = tensor_a.shape_;
    const auto& shape_b = tensor_b.shape_;

    if (shape_a.size() < 2 || shape_b.size() < 2) {
        std::cerr << "MatMul: inputs must have at least 2 dimensions" << std::endl;
        return false;
    }

    int64_t k_a = shape_a[shape_a.size() - 1];
    int64_t k_b = shape_b[shape_b.size() - 2];

    if (k_a > 0 && k_b > 0 && k_a != k_b) {
        std::cerr << "MatMul dimension mismatch: K_a=" << k_a << " vs K_b=" << k_b << std::endl;
        return false;
    }

    if (shape_a.size() > 2 || shape_b.size() > 2) {
        size_t rank_a = shape_a.size();
        size_t rank_b = shape_b.size();
        size_t max_rank = std::max(rank_a, rank_b);

        for (size_t i = 0; i < max_rank - 2; ++i) {
            int64_t dim_a = (i < rank_a - 2) ? shape_a[rank_a - 3 - i] : 1;
            int64_t dim_b = (i < rank_b - 2) ? shape_b[rank_b - 3 - i] : 1;


            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                std::cerr << "MatMul broadcast mismatch at batch dim " << i
                          << ": " << dim_a << " vs " << dim_b << std::endl;
                return false;
            }
        }
    }

    return true;
}
bool OpMaxPool::verify(const Core::BebraGraph& graph) const {
    return true;
}
bool OpReduceMean::verify(const Core::BebraGraph& graph) const {

        const auto& tensor_in = graph.getTensor(input);

        if (!tensor_in.hasShape()) {
            return true;
        }

        const auto& shape = tensor_in.shape_;
        int64_t num_dims = static_cast<int64_t>(shape.size());


        for (int64_t axis : axes) {

            int64_t normalized_axis = axis < 0 ? axis + num_dims : axis;

            if (normalized_axis < 0 || normalized_axis >= num_dims) {
                std::cerr << "Reduce axis " << axis << " out of bounds for tensor with "
                          << num_dims << " dimensions" << std::endl;
                return false;
            }
        }


        if (keepdims != 0 && keepdims != 1) {
            std::cerr << "ReduceMean keepdims must be 0 or 1, got " << keepdims << std::endl;
            return false;
        }

        return true;
}
bool OpReshape::verify(const Core::BebraGraph& graph) const {
        return true;
}
bool OpSigmoid::verify(const Core::BebraGraph& graph) const {

        const auto& tensor_in = graph.getTensor(input);

        if (!tensor_in.hasShape()) {
            return true;
        }

        for (int64_t dim : tensor_in.shape_) {
            if (dim <= 0) {
                std::cerr << "Invalid dimension size " << dim << std::endl;
                return false;
            }
        }

        return true;
}
bool OpFlatten::verify(const Core::BebraGraph& graph) const {
        return true;
}
std::vector<int64_t> OpVoid::countOutputShape(Core::BebraGraph& graph) {    return {};
}
std::vector<int64_t> OpConv::countOutputShape(Core::BebraGraph& graph) {    auto inputShape = graph.getTensor(input).getShape();
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
}
std::vector<int64_t> OpGemm::countOutputShape(Core::BebraGraph& graph) {    return {};
}
std::vector<int64_t> OpAdd::countOutputShape(Core::BebraGraph& graph) {    auto shapeA = graph.getTensor(input_1).getShape();
    auto shapeB = graph.getTensor(input_2).getShape();
    return calculateBroadcastShape(shapeA, shapeB);
}
std::vector<int64_t> OpRelu::countOutputShape(Core::BebraGraph& graph) {return graph.getTensor(input).getShape();}
std::vector<int64_t> OpMul::countOutputShape(Core::BebraGraph& graph) {    auto shapeA = graph.getTensor(input_1).getShape();
    auto shapeB = graph.getTensor(input_2).getShape();
    return calculateBroadcastShape(shapeA, shapeB);
}
std::vector<int64_t> OpMatMul::countOutputShape(Core::BebraGraph& graph) {    auto shapeA = graph.getTensor(input_a).getShape();
    auto shapeB = graph.getTensor(input_b).getShape();
    // [M, K] x [K, N] -> [M, N]
    return calculateMatMulShape(shapeA, shapeB);
}
std::vector<int64_t> OpMaxPool::countOutputShape(Core::BebraGraph& graph) {    if (kernel_shape.size() != 2) {
        throw Core::BebraErr("only 2D MaxPool is supported for now!");
    }


    return {};
}
std::vector<int64_t> OpReduceMean::countOutputShape(Core::BebraGraph& graph) {    auto inputShape = graph.getTensor(input).getShape();
    return calculateReduceShape(inputShape, axes, keepdims);
}
std::vector<int64_t> OpReshape::countOutputShape(Core::BebraGraph& graph) {return graph.getTensor(shape).getShape();}
std::vector<int64_t> OpSigmoid::countOutputShape(Core::BebraGraph& graph) {return graph.getTensor(input).getShape();}
std::vector<int64_t> OpFlatten::countOutputShape(Core::BebraGraph& graph) {return calculateFlattenShape(graph.getTensor(input).getShape(), axis);}
}
