#pragma once

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include <optional>

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"


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
    void generate(const Core::BebraGraph& graph, std::string& out_str);
    void dump(const std::string& filename, const std::string& dumped);

public: // mlir-specific methods
    mlir::OwningOpRef<mlir::ModuleOp> createModule();
    std::optional<mlir::Value> getSSA(const std::string& val_name) {
        auto&& val = ssa_map_.find(val_name);

        if (val != ssa_map_.end()) {
            return std::optional<mlir::Value>(val->second);
        }

        return std::nullopt;
        // throw Core::BebraErr("can't find ssa with such name " + val_name + "!");
    }

    void setSSA(const std::string& val_name, mlir::Value val) {
        // // std::cout << "setting " << val_name << "\n";
        // // std::cout << val << "\n";
        Core::BebraGreen("Setting ssa " + val_name);
        ssa_map_[val_name] = std::move(val);
    }

    std::optional<mlir::Type> getType(const std::string& name) {
        auto found = type_map_.find(name);
        if (found != type_map_.end()) {
            return std::optional<mlir::Type>(found->second);
        }

        return std::nullopt;
    }

private: // mlir-specific
    void loadAllNeededDialects();
    mlir::RankedTensorType createTensorType(const Core::BebraTensor& tensor);
    mlir::RankedTensorType createDynamicTensorType(mlir::Value& tensor);
    mlir::Value createFilledTensor(mlir::RankedTensorType type, mlir::Value sourceTensor);
    mlir::Type getElementType(Core::BebraType type);
    mlir::LogicalResult compileToLLVM(mlir::ModuleOp module, llvm::raw_string_ostream& stream);
    mlir::DenseI64ArrayAttr getDenseI64ArrayAttrFromValue(mlir::Value value);

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
