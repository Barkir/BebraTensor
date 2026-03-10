#include "bebra/mlir/MLIRHelpers.hpp"
#include "onnx_proto/onnx.proto3.pb.h"


mlir::OwningOpRef<mlir::ModuleOp> createModule(const onnx::GraphProto& graph) {
    mlir::MLIRContext context;
    mlir::OpBuilder builder(&context);

    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());



    return module;

}
