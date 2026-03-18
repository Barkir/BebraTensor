#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

#include "bebra/core/BebraColors.hpp"
#include "bebra/core/BebraErr.hpp"
#include "bebra/core/BebraGraph.hpp"
#include "bebra/core/BebraTensor.hpp"
#include "bebra/core/BebraType.hpp"
#include "bebra/mlir/MLIRPrinter.hpp"

#include <iostream>

namespace Bebra::MLIR {

void MLIRPrinter::Visit(const Ops::OpVoid& node) {
    std::cout << "void" << "\n";
}

void MLIRPrinter::Visit(const Ops::OpConv& node) {
    std::cout << "conv" << "\n";

    // inputs
    auto input = getSSA(node.input);
    if (!input) {
        Core::BebraWarn("no input provided + " + node.input);
    }

    auto bias = getSSA(node.bias);
    if (!bias) {
        Core::BebraWarn("no bias provided + " + node.bias);
    }

    auto weight = getSSA(node.weight);
    if (!weight) {
        Core::BebraWarn("no weight provided + " + node.weight);
    }

    std::cout << "got ssa\n";

    // attrs
    auto kernel_shape = node.kernel_shape;
    if (kernel_shape.size() != 2) {
        throw Core::BebraErr("only 2D convolution is supported in linalg dialect");
    }

    auto strides = node.strides;
    auto pads = node.pads;
    int64_t group = node.group;
    if (group != 1) {
        throw Core::BebraErr("only group = 1 is supported in linalg dialect");
    }
    auto dilations = node.dilations;

    auto stridesDenseAttr = builder_.getI64TensorAttr(strides);
    auto dilationsDenseAttr = builder_.getI64TensorAttr(dilations);
    std::cout << "got tensor attrs for strides and dilations for builder\n";

    auto outtype = createDynamicTensorType(*input);
    auto filledTensor = createFilledTensor(outtype);

    // std::cout << filledTensor << "\n";
    // counting output
    auto output = builder_.create<mlir::linalg::Conv2DNchwFchwOp>(builder_.getUnknownLoc(),
                                                                  outtype,
                                                                  mlir::ValueRange{*input, *weight},
                                                                  mlir::ValueRange{filledTensor}, // FIXME -
                                                                  stridesDenseAttr,
                                                                  dilationsDenseAttr

    ).getResult(0);
    std::cout << "created output\n";

    setSSA(node.output, output);
    std::cout << "visited conv" << "\n";
}

void MLIRPrinter::Visit(const Ops::OpGemm& node) {
    std::cout << "gemm" << "\n";

    // inputs
    auto input_a = getSSA(node.input_a);
    if (!input_a) {
        Core::BebraWarn("No input_a in gemm: " + node.input_a);
    }
    auto input_b = getSSA(node.input_b);
    if (!input_b) {
        Core::BebraWarn("No input_a in gemm: " + node.input_a);
    }

    // attrs

    // output
    // auto output =
}

void MLIRPrinter::Visit(const Ops::OpAdd& node) {
    std::cout << "add" << "\n";
    auto lhs = getSSA(node.input_1);
    if (!lhs) {
        Core::BebraWarn("no lhs provided in OpAdd: " + node.input_1);
    }
    auto rhs = getSSA(node.input_2);
    if (!rhs) {
        Core::BebraWarn("no rhs provided in OpAdd:" + node.input_2);
    }

    auto fastmath = mlir::arith::FastMathFlagsAttr::get(builder_.getContext(), mlir::arith::FastMathFlags::none);

    mlir::Type type = (*lhs).getType();

    mlir::Value output = builder_.create<mlir::arith::AddFOp>(builder_.getUnknownLoc(), type, *lhs, *rhs, fastmath);

    setSSA(node.output, output);
    std::cerr << "visited add" << "\n";
}

void MLIRPrinter::Visit(const Ops::OpRelu& node) {
    std::cout << "relu" << std::endl;

    auto input = getSSA(node.input);
    if (!input) {
        Core::BebraWarn("no input at relu: " + node.input);
    }

    auto outtype = createDynamicTensorType(*input);

    mlir::Value zero = builder_.create<mlir::arith::ConstantOp>(
    builder_.getUnknownLoc(),
    outtype.getElementType(),
    builder_.getZeroAttr(outtype.getElementType()));


    auto output = builder_.create<mlir::arith::MaximumFOp>(
        builder_.getUnknownLoc(),
        outtype,
        zero,
        *input
    );

    setSSA(node.output, output);
    std::cout << "visited relu" << "\n";
}

void MLIRPrinter::Visit(const Ops::OpMul& node) {
    std::cout << "mul" << "\n";
    auto lhs = getSSA(node.input_1);
    if (!lhs) {
        Core::BebraWarn("no lhs provided in OpMul: " + node.input_1);
    }
    auto rhs = getSSA(node.input_2);
    if (!rhs) {
        Core::BebraWarn("no rhs provided in OpMul:" + node.input_2);
    }

    auto fastmath = mlir::arith::FastMathFlagsAttr::get(builder_.getContext(), mlir::arith::FastMathFlags::none);

    mlir::Type type = (*lhs).getType();

    mlir::Value output = builder_.create<mlir::arith::MulFOp>(builder_.getUnknownLoc(), type, *lhs, *rhs, fastmath);

    setSSA(node.output, output);
    std::cerr << "visited mul" << "\n";
}

void MLIRPrinter::Visit(const Ops::OpMatMul& node) {
    std::cout << "matmul" << "\n";

    auto input_a = getSSA(node.input_a);
    if (!input_a) {
        Core::BebraWarn("can't get input_a: " + node.input_a);
    }

    auto input_b = getSSA(node.input_b);
    if (!input_b) {
        Core::BebraWarn("can't get input_b: " + node.input_b);
    }

    auto outtype = createDynamicTensorType(*input_a);
    mlir::Value filledTensor = createFilledTensor(outtype);

    auto output = builder_.create<mlir::linalg::MatmulOp>(
        builder_.getUnknownLoc(),
        mlir::ValueRange{*input_a, *input_b},
        mlir::ValueRange{filledTensor}
    ).getResult(0);

    setSSA(node.output, output);
    std::cout << "visited matmul" << "\n";
}

void MLIRPrinter::Visit(const Ops::OpMaxPool& node) {
    std::cout << "maxpool" << " \n";
    auto input = getSSA(node.input);
    if (!input) {
        Core::BebraWarn("no input at maxpool: " + node.input);
    }


    // attrs
    auto kernel_shape = node.kernel_shape;
    if (kernel_shape.size() != 2) {
        throw Core::BebraErr("Only 2D MaxPool is supported now...");
    }
    std::string auto_pad = node.auto_pad;
    int64_t ceil_mode = node.ceil_mode;
    std::vector<int64_t> dilations = node.dilations;
    std::vector<int64_t> pads = node.pads;
    int64_t storage_order = node.storage_order;
    std::vector<int64_t> strides = node.strides;

    std::cout << "creating dilations attrs..." << "\n";
    auto kernelAttr = builder_.getDenseI64ArrayAttr(kernel_shape);
    auto dilationsAttr = builder_.getDenseI64ArrayAttr(dilations);
    auto stridesAttr = builder_.getDenseI64ArrayAttr(strides);
    auto padsAttr = builder_.getDenseI64ArrayAttr(pads);

    auto outtype = createDynamicTensorType(*input);
    mlir::Value filledTensor = createFilledTensor(outtype);
    // output
    std::cout << "creating output" << std::endl;
    auto output = builder_.create<mlir::tosa::MaxPool2dOp>(
        builder_.getUnknownLoc(),
        outtype,
        *input,
        kernelAttr,
        stridesAttr,
        padsAttr
    );

    setSSA(node.output, output);
    std::cout << "visited maxpool" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpReduceMean& node) {
    std::cout << "reducemean" << "\n";
}

void MLIRPrinter::Visit(const Ops::OpReshape& node) {
    std::cout << "reshape" << "\n";
    auto input = getSSA(node.input);
    if (!input) {
        Core::BebraWarn("no input at reshape: " + node.input);
    }
    auto shape = getSSA(node.shape);
    if (!shape) {
        Core::BebraWarn("no shape at reshape: " + node.shape);
    }

    auto outtype = createDynamicTensorType(*shape);

    auto output = builder_.create<mlir::tosa::ReshapeOp>(
        builder_.getUnknownLoc(),
        outtype,
        mlir::ValueRange{*input, *shape}
    );

    setSSA(node.output, output);
    std::cout << "visited reshape" << "\n";
}

void MLIRPrinter::Visit(const Ops::OpSigmoid& node) {
    std::cout << "sigmoid" << "\n";
}

void MLIRPrinter::Visit(const Ops::OpFlatten& node) {
    std::cout << "flatten" << "\n";
}

void MLIRPrinter::Visit(const Core::BebraTensor& tensor) {
    auto name = tensor.getName();
    std::cout << "got tensor name " << name << "\n";
    auto ssa = getSSA(name);

    if (!ssa) {
        Core::BebraWarn("SSA not set for tensor " + name);
        mlir::RankedTensorType ttype = createTensorType(tensor);

        auto& data = tensor.data();

        // std::cout << "tensor of type := " << ttype << " // data size := " << data.size() << "\n";
        auto denseAttr = mlir::DenseElementsAttr::get(ttype, llvm::ArrayRef(data));
        denseAttr.dump();

        mlir::Value ssa_val = builder_.create<mlir::arith::ConstantOp>(builder_.getUnknownLoc(), denseAttr);
        setSSA(name, ssa_val);
        // // std::cout << "visited tensor " << name << " // -> " << ssa_val << "\n";
        return;
    }

    // // std::cout << "visited tensor " << name << ssa << "\n";
    return;
}

// ====================================================================================

MLIRPrinter::MLIRPrinter(Core::BebraGraph& graph) : builder_(&context_) {
    for (auto&& tensor : graph.tensor_map_) {
        auto&& tname = tensor.first;
        auto&& thetensor = tensor.second;
        auto&& type = createTensorType(thetensor);
        type_map_[tname] = type;
    }
}

void MLIRPrinter::dump(const std::string& filename, const std::string& dumped) {
    std::ofstream os(filename);
    os << dumped;

}


void MLIRPrinter::generate(const Core::BebraGraph& graph, std::string& out_str) {
    loadAllNeededDialects();
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
    builder_.setInsertionPointToStart(module.getBody());

    // initializing start SSA-values
    // this can be : inputs, weights and other stuff that we
    // need to init a neural network

    // ==================================================================

    for (auto&& initializer_name : graph.initializers_) {
        auto&& initializer = graph.getTensor(initializer_name);
        Visit(initializer);
    }

    for (auto&& input_name : graph.inputs_) {
        std::cout << "VISITING INPUTS" << "\n";
        auto&& input = graph.getTensor(input_name);
        Visit(input);
    }

    // ==================================================================
    std::cout << "VISITING NODES" << "\n";
    for (auto&& node : graph.nodes_) {
        std::visit([this](auto&& op) { Visit(op); }, node.op_);
    }

    llvm::raw_string_ostream ss(out_str);

    mlir::OpPrintingFlags flags;
    module.print(ss, flags);
}

// ================= mlir-specific ==========================

void MLIRPrinter::loadAllNeededDialects() {
    context_.loadDialect<mlir::arith::ArithDialect>();
    context_.loadDialect<mlir::tensor::TensorDialect>();
    context_.loadDialect<mlir::func::FuncDialect>();
    context_.loadDialect<mlir::linalg::LinalgDialect>();
    context_.loadDialect<mlir::tosa::TosaDialect>();












































}

mlir::Type MLIRPrinter::getElementType(Core::BebraType type) {
    mlir::Type elementType;
    switch (type) {
        case Core::BebraType::FLOAT:
            elementType = builder_.getF32Type();
            break;
        case Core::BebraType::DOUBLE:
            elementType = builder_.getF64Type();
            break;
        case Core::BebraType::INT32:
            elementType = builder_.getI32Type();
            break;
        case Core::BebraType::INT64:
            elementType = builder_.getI64Type();
            break;
        case Core::BebraType::BOOL:
            elementType = builder_.getI8Type();
            break;
        case Core::BebraType::INT8:
            elementType = builder_.getI8Type();
            break;
        case Core::BebraType::UINT8:
            elementType = builder_.getI8Type();
            break;
        default:
            elementType = builder_.getF32Type();
    }
    return elementType;
}

mlir::RankedTensorType MLIRPrinter::createTensorType(const Core::BebraTensor& tensor) {
    mlir::Type elementType = getElementType(tensor.dtype);
    return mlir::RankedTensorType::get(tensor.getShape(), elementType);
}


mlir::RankedTensorType MLIRPrinter::createDynamicTensorType(mlir::Value& tensor) {
    auto ttype = mlir::dyn_cast<mlir::RankedTensorType>(tensor.getType());
    if (!ttype) {
        throw Core::BebraErr("Value is not a RankedTensor while creating dynamic tensor...");
    }
    auto ndims = ttype.getRank();
    auto eltype = ttype.getElementType();;
    llvm::SmallVector<int64_t> shape(ndims, mlir::ShapedType::kDynamic);
    return mlir::RankedTensorType::get(shape, eltype);
}
mlir::Value MLIRPrinter::createFilledTensor(mlir::RankedTensorType& type) {
    mlir::Value emptyTensor = builder_.create<mlir::tensor::EmptyOp>(
        builder_.getUnknownLoc(),
        type.getShape(),
        type.getElementType());

    mlir::Value zero = builder_.create<mlir::arith::ConstantOp>(
        builder_.getUnknownLoc(),
        type.getElementType(),
        builder_.getZeroAttr(type.getElementType()));

    mlir::Value filledTensor = builder_.create<mlir::linalg::FillOp>(
        builder_.getUnknownLoc(),
        mlir::ValueRange{zero},
        mlir::ValueRange{emptyTensor}).getResult(0);

    return filledTensor;
}

} // namespace Bebra::MLIR
