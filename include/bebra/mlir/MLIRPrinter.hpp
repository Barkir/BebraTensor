#pragma once

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include <optional>

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

#include "bebra/core/BebraErr.hpp"
#include "bebra/ops/BebraOperators.hpp"
#include "bebra/core/BebraLog.hpp"
#include "bebra/mlir/ModifiedValue.hpp"

namespace Bebra::Core {
class BebraGraph;
class BebraTensor;
} // namespace Bebra::Core

namespace Bebra::MLIR {

class MLIRPrinter {

    mlir::DialectRegistry registry;
    mlir::MLIRContext context_;
    mlir::OpBuilder builder_;
    std::unordered_map<std::string, mlir::Value> ssa_map_;
    std::unordered_map<std::string, ModifiedValue> type_map_;


public: // constructor
    MLIRPrinter(Core::BebraGraph& graph);

    ~MLIRPrinter() = default;

public: // helpers
    void generate(const Core::BebraGraph& graph, std::string& out_str);
    void dump(const std::string& filename, const std::string& dumped);

public: // mlir-specific methods
    mlir::OwningOpRef<mlir::ModuleOp> createModule();
    std::optional<mlir::Value> getSSA(const std::string& val_name) {
        LOG("Getting SSA of name: {}\n", val_name);
        auto&& val = ssa_map_.find(val_name);

        if (val != ssa_map_.end()) {
            LOG("Found SSA of name {}\n", val_name);
            return std::optional<mlir::Value>(val->second);
        }

        LOG("Not found SSA of name {}\n", val_name);
        return std::nullopt;
        // throw Core::BebraErr("can't find ssa with such name " + val_name + "!");
    }

    void setSSA(const std::string& val_name, mlir::Value val) {
        Core::BebraGreen("Setting ssa " + val_name);
        ssa_map_[val_name] = std::move(val);
    }

    std::string getNameBySSA(const mlir::Value& val) {
        MSG("Getting name by ssa...\n");
        for (auto&& iter : ssa_map_) {
            if (iter.second == val) {
                Core::BebraWarn("Got name by ssa <<<<< (:");
                return iter.first;
            }
        }

        return "";
    }

// ---------------------------------------------------------------------------------

    void setType(const std::string& val_name, mlir::Type type) {
        ModifiedValue value;
        value.type = type;
        value.dstype = DataStoreType::UNDEFINED;

        auto found = type_map_.find(val_name);
        if (found == type_map_.end()) {
            type_map_.emplace(val_name, value);
            return;
        }

        auto new_val = found->second;
        new_val.type = type;
        type_map_.insert_or_assign(val_name, new_val);
        return;
    }

    void setStoreType(const std::string& val_name, DataStoreType type) {
        ModifiedValue value;
        LOG("Setting store type of {} to {}\n", val_name, static_cast<int>(type));
        value.dstype = type;
        auto found = type_map_.find(val_name);
        if (found == type_map_.end()) {
            type_map_.insert_or_assign(val_name, value);
            return;
        }

        auto new_val = found->second;
        new_val.dstype = type;
        type_map_.insert_or_assign(val_name, new_val);
        return;
    }

// -------------------------------------------------------------------------------

    DataStoreType getStoreType(const std::string& name) {
        if (name == "") {
            LOG("{} got UNDEFINED STORE TYPE\n", name);
            return DataStoreType::UNDEFINED;
        }
        auto found = type_map_.find(name);
        if (found != type_map_.end()) {
            LOG("{} got {} store type\n", name, static_cast<int>(found->second.dstype));
            return (found->second.dstype);
        }

        LOG("{} got undefined store type\n", name);
        return DataStoreType::UNDEFINED;
    }

    std::optional<mlir::Type> getType(const std::string& name) {
        auto found = type_map_.find(name);
        if (found != type_map_.end()) {
            return std::optional<mlir::Type>(found->second.type);
        }

        return std::nullopt;
    }

private: // mlir-specific
    void loadAllNeededDialects();
    mlir::RankedTensorType createTensorType(const Core::BebraTensor& tensor);
    mlir::RankedTensorType createDynamicTensorType(mlir::Value& tensor);
    mlir::RankedTensorType createDynamicTensorType(size_t ndims, mlir::Type eltype);
    mlir::RankedTensorType createDynamicTensorType(mlir::Value& tensor, size_t ndims);
    mlir::UnrankedTensorType createUnrankedTensorType(mlir::Type eltype);
    mlir::UnrankedTensorType createUnrankedTensorType(mlir::Value& tensor);
    mlir::Value createFilledTensor(mlir::RankedTensorType type);
    mlir::Type getElementType(Core::BebraType type);
    mlir::LogicalResult compileToLLVM(mlir::ModuleOp module, llvm::raw_string_ostream& stream);
    mlir::DenseI64ArrayAttr getDenseI64ArrayAttrFromValue(mlir::Value value);
    mlir::RankedTensorType broadCastType(mlir::RankedTensorType type, size_t toRank);

    mlir::RankedTensorType computeMaxPool2DOutputType (mlir::Value input, mlir::DenseArrayAttr kernel_size,
                                                                    mlir::DenseArrayAttr stride, mlir::DenseArrayAttr pad,
                                                                    mlir::DenseArrayAttr dilation);

    mlir::RankedTensorType computeConv2DOutputType(mlir::Value input, mlir::Value weight,
                                                                mlir::DenseArrayAttr kernel,
                                                                mlir::DenseArrayAttr stride,
                                                                mlir::DenseArrayAttr pad,
                                                                mlir::DenseArrayAttr dilation,
                                                                mlir::Type accType);

    mlir::RankedTensorType computeMatMulOutputType(mlir::Value& input_a,
                                                   mlir::Value& input_b);

    llvm::SmallVector<mlir::Value> collectReturnValues(const Core::BebraGraph& graph);



    mlir::Value convertNCHWToNHWCVal(mlir::Value& val);
    mlir::RankedTensorType convertNCHWToNHWC(mlir::Type type);
    llvm::SmallVector<mlir::Type> createInputTypes(const Core::BebraGraph& graph);
    llvm::SmallVector<mlir::Type> createOutputTypes(const Core::BebraGraph& graph);

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
