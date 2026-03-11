#pragma once

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "bebra/core/BebraErr.hpp"
#include "bebra/ops/BebraOperators.hpp"

namespace Bebra::Core { class BebraGraph; class BebraTensor; } // forward declaration

namespace Bebra::MLIR {

class MLIRPrinter {
	mlir::MLIRContext context_;
	mlir::OpBuilder builder_;
	std::unordered_map<std::string, mlir::Value*> ssa_map_;
	std::unordered_map<std::string, mlir::Type> type_map_;


public: // constructor
	MLIRPrinter(Core::BebraGraph& graph);

	~MLIRPrinter() = default;


public: // helpers
	std::string generate(const Core::BebraGraph& graph);

public: // mlir-specific methods
	mlir::OwningOpRef<mlir::ModuleOp> createModule();
	mlir::Value* getSSA(std::string& val_name) {
		auto&& val = ssa_map_.find(val_name);

		if (val != ssa_map_.end()) {
			return val->second;
		}

		throw Core::BebraErr("can't find ssa with such name " + val_name + "!");

	}

	mlir::Type createTensorType(const Core::BebraTensor& tensor);


public: // Visitors
	void Visit(const Ops::OpVoid& node) const;
	void Visit(const Ops::OpConv& node) const;
	void Visit(const Ops::OpGemm& node) const;
	void Visit(const Ops::OpAdd& node) const;
	void Visit(const Ops::OpRelu& node) const;
	void Visit(const Ops::OpMul& node) const;
	void Visit(const Ops::OpMatMul& node) const;
	void Visit(const Ops::OpMaxPool& node) const;
	void Visit(const Ops::OpReduceMean& node) const;
	void Visit(const Ops::OpReshape& node) const;
	void Visit(const Ops::OpSigmoid& node) const;
	void Visit(const Ops::OpFlatten& node) const;
	void Visit(const Core::BebraTensor& tensor) const;

};



}
