#pragma once

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "bebra/core/BebraErr.hpp"
#include "bebra/ops/BebraOperators.hpp"

namespace Bebra::Core {
class BebraGraph;
class BebraTensor;
} // namespace Bebra::Core

namespace Bebra::MLIR {

class MLIRPrinter {
    mlir::MLIRContext context_;
    mlir::OpBuilder builder_;
    std::unordered_map<std::string, mlir::Value> ssa_map_;
    std::unordered_map<std::string, mlir::Type> type_map_;

public: // constructor
    MLIRPrinter(Core::BebraGraph& graph);

    ~MLIRPrinter() = default;

public: // helpers
    std::string generate(const Core::BebraGraph& graph);

public: // mlir-specific methods
    mlir::OwningOpRef<mlir::ModuleOp> createModule();
    mlir::Value getSSA(const std::string& val_name) {
        auto&& val = ssa_map_.find(val_name);

        if (val != ssa_map_.end()) {
            return val->second;
        }

        return {};
        throw Core::BebraErr("can't find ssa with such name " + val_name + "!");
    }

    void setSSA(const std::string& val_name, mlir::Value val) {
        ssa_map_[val_name] = std::move(val);
        Core::BebraGreen("Setting ssa " + val_name);
    }

private: // mlir-specific

    void loadAllNeededDialects();
    mlir::RankedTensorType createTensorType(const Core::BebraTensor& tensor);

private: // Visitors
    void Visit(const Ops::OpVoid& node);
    void Visit(const Ops::OpConv& node);
    void Visit(const Ops::OpGemm& node);
    void Visit(const Ops::OpAdd& node);
    void Visit(const Ops::OpRelu& node);
    void Visit(const Ops::OpMul& node);
    void Visit(const Ops::OpMatMul& node);
    void Visit(const Ops::OpMaxPool& node);
    void Visit(const Ops::OpReduceMean& node);
    void Visit(const Ops::OpReshape& node);
    void Visit(const Ops::OpSigmoid& node);
    void Visit(const Ops::OpFlatten& node);
    void Visit(const Core::BebraTensor& tensor);
};

} // namespace Bebra::MLIR
