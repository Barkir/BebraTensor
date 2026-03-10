#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"

#include "bebra/mlir/MLIRPrinter.hpp"
#include "bebra/core/BebraGraph.hpp"
#include <iostream>

namespace Bebra::MLIR {

	void MLIRPrinter::Visit(const Ops::OpVoid& node)          const { std::cout << "void" << std::endl; }
	void MLIRPrinter::Visit(const Ops::OpConv& node)          const { std::cout << "conv" << std::endl; }
	void MLIRPrinter::Visit(const Ops::OpGemm& node)          const { std::cout << "gemm" << std::endl; }
	void MLIRPrinter::Visit(const Ops::OpAdd& node)           const { std::cout << "add" << std::endl; }
	void MLIRPrinter::Visit(const Ops::OpRelu& node)          const { std::cout << "relu" << std::endl; }
	void MLIRPrinter::Visit(const Ops::OpMul& node)           const { std::cout << "mul" << std::endl; }
	void MLIRPrinter::Visit(const Ops::OpMatMul& node)        const { std::cout << "matmul" << std::endl; }
	void MLIRPrinter::Visit(const Ops::OpMaxPool& node)       const { std::cout << "maxpool" << std::endl; }
	void MLIRPrinter::Visit(const Ops::OpReduceMean& node)    const { std::cout << "reducemean" << std::endl; }
	void MLIRPrinter::Visit(const Ops::OpReshape& node)       const { std::cout << "reshape" << std::endl; }
	void MLIRPrinter::Visit(const Ops::OpSigmoid& node)       const { std::cout << "sigmoid" << std::endl; }
	void MLIRPrinter::Visit(const Ops::OpFlatten& node)       const { std::cout << "flatten" << std::endl; }

std::string MLIRPrinter::generate(const Core::BebraGraph& graph) {
    for (auto&& node : graph.nodes_) {
        std::visit([this](auto&& op) {
            Visit(op);
        }, node.op_);
    }

    return {};
}
}
