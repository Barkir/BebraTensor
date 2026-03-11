#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"

#include "bebra/mlir/MLIRPrinter.hpp"
#include "bebra/core/BebraGraph.hpp"
#include "bebra/core/BebraTensor.hpp"
#include "bebra/core/BebraType.hpp"

#include <iostream>

namespace Bebra::MLIR {

void MLIRPrinter::Visit(const Ops::OpVoid& node)          const { std::cout << "void" << std::endl; }

void MLIRPrinter::Visit(const Ops::OpConv& node)          const { std::cout << "conv" << std::endl; }

void MLIRPrinter::Visit(const Ops::OpGemm& node)          const { std::cout << "gemm" << std::endl; }

void MLIRPrinter::Visit(const Ops::OpAdd& node)           const {
	std::cout << "add" << std::endl;
}

void MLIRPrinter::Visit(const Ops::OpRelu& node)          const { std::cout << "relu" << std::endl; }

void MLIRPrinter::Visit(const Ops::OpMul& node)           const { std::cout << "mul" << std::endl; }

void MLIRPrinter::Visit(const Ops::OpMatMul& node)        const { std::cout << "matmul" << std::endl; }

void MLIRPrinter::Visit(const Ops::OpMaxPool& node)       const { std::cout << "maxpool" << std::endl; }

void MLIRPrinter::Visit(const Ops::OpReduceMean& node)    const { std::cout << "reducemean" << std::endl; }

void MLIRPrinter::Visit(const Ops::OpReshape& node)       const { std::cout << "reshape" << std::endl; }

void MLIRPrinter::Visit(const Ops::OpSigmoid& node)       const { std::cout << "sigmoid" << std::endl; }

void MLIRPrinter::Visit(const Ops::OpFlatten& node)       const { std::cout << "flatten" << std::endl; }

void MLIRPrinter::Visit(const Core::BebraTensor& tensor)  const {

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
    for (auto&& node : graph.nodes_)
        std::visit([this](auto&& op) {
            Visit(op);
        }, node.op_);

    return "";
}



// ================= mlir-specific ==========================

mlir::Type MLIRPrinter::createTensorType(const Core::BebraTensor& tensor) {
    mlir::Type elementType;
    switch (tensor.dtype) {
        case Core::BebraType::FLOAT:      elementType = builder_.getF32Type(); break;
        case Core::BebraType::DOUBLE:     elementType = builder_.getF64Type(); break;
        case Core::BebraType::INT32:      elementType = builder_.getI32Type(); break;
        case Core::BebraType::INT64:      elementType = builder_.getI64Type(); break;
        case Core::BebraType::BOOL:       elementType = builder_.getI8Type();  break;
        case Core::BebraType::INT8:       elementType = builder_.getI8Type();  break;
        case Core::BebraType::UINT8:      elementType = builder_.getI8Type();  break;
        default:                          elementType = builder_.getF32Type();
    }

    return mlir::RankedTensorType::get(tensor.getShape(), elementType);
}

}
