#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
    std::cout << "void" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpConv& node) {
    std::cout << "conv" << std::endl;

    // inputs
    auto input = getSSA(node.input);
    auto bias = getSSA(node.bias);
    auto weight = getSSA(node.weight);
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

    // counting output
    auto output = builder_.create<mlir::linalg::Conv2DNchwFchwOp>(builder_.getUnknownLoc(),
                                                                  mlir::TypeRange{}, // FIXME - count output type
                                                                  mlir::ValueRange{input, weight},
                                                                  mlir::ValueRange{}, // FIXME -
                                                                  stridesDenseAttr,
                                                                  dilationsDenseAttr

    );
    std::cout << "created output\n";

    // setSSA(node.output, output);
    std::cout << "visited conv" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpGemm& node) {
    std::cout << "gemm" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpAdd& node) {
    std::cout << "add" << std::endl;
    auto lhs = getSSA(node.input_1);
    auto rhs = getSSA(node.input_2);

    auto fastmath = mlir::arith::FastMathFlagsAttr::get(builder_.getContext(), mlir::arith::FastMathFlags::none);

    mlir::Type type = lhs.getType();

    mlir::Value output = builder_.create<mlir::arith::AddFOp>(builder_.getUnknownLoc(), type, lhs, rhs, fastmath);

    setSSA(node.output, output);
    std::cerr << "visited add" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpRelu& node) {
    std::cout << "visited relu" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpMul& node) {
    auto lhs = getSSA(node.input_1);
    auto rhs = getSSA(node.input_2);

    auto fastmath = mlir::arith::FastMathFlagsAttr::get(builder_.getContext(), mlir::arith::FastMathFlags::none);

    mlir::Type type = lhs.getType();

    mlir::Value output = builder_.create<mlir::arith::MulFOp>(builder_.getUnknownLoc(), type, lhs, rhs, fastmath);

    setSSA(node.output, output);
    std::cerr << "visited mul" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpMatMul& node) {
    std::cout << "matmul" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpMaxPool& node) {
    std::cout << "maxpool" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpReduceMean& node) {
    std::cout << "reducemean" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpReshape& node) {
    std::cout << "reshape" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpSigmoid& node) {
    std::cout << "sigmoid" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpFlatten& node) {
    std::cout << "flatten" << std::endl;
}

void MLIRPrinter::Visit(const Core::BebraTensor& tensor) {
    auto name = tensor.getName();
    auto ssa = getSSA(name);
    if (!ssa) {
        Core::BebraWarn("SSA not set for tensor " + name);
        mlir::RankedTensorType ttype = createTensorType(tensor);

        auto& data = tensor.data();

        llvm::outs() << "tensor of type := " << ttype << " // data size := " << data.size() << "\n";
        auto denseAttr = mlir::DenseElementsAttr::get(ttype, llvm::ArrayRef(data));
        denseAttr.dump();

        mlir::Value ssa_val = builder_.create<mlir::arith::ConstantOp>(builder_.getUnknownLoc(), denseAttr);
        setSSA(name, ssa_val);
        llvm::outs() << "visited tensor " << name << " // -> " << ssa_val << "\n";
        return;
    }
    llvm::outs() << "visited tensor " << name << ssa << "\n";
    return;
}

MLIRPrinter::MLIRPrinter(Core::BebraGraph& graph) : builder_(&context_) {
    for (auto&& tensor : graph.tensor_map_) {
        auto&& tname = tensor.first;
        auto&& thetensor = tensor.second;
        auto&& type = createTensorType(thetensor);
        type_map_[tname] = type;
    }
}

std::string MLIRPrinter::generate(const Core::BebraGraph& graph) {
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
        auto&& input = graph.getTensor(input_name);
        Visit(input);
    }

    // ==================================================================

    for (auto&& node : graph.nodes_) {
        std::visit([this](auto&& op) { Visit(op); }, node.op_);
    }

    module.dump();
    return "";
}

// ================= mlir-specific ==========================

void MLIRPrinter::loadAllNeededDialects() {
    context_.loadDialect<mlir::arith::ArithDialect>();
    context_.loadDialect<mlir::tensor::TensorDialect>();
    context_.loadDialect<mlir::func::FuncDialect>();
    context_.loadDialect<mlir::linalg::LinalgDialect>();
}

mlir::RankedTensorType MLIRPrinter::createTensorType(const Core::BebraTensor& tensor) {
    mlir::Type elementType;
    switch (tensor.dtype) {
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

    return mlir::RankedTensorType::get(tensor.getShape(), elementType);
}

} // namespace Bebra::MLIR
