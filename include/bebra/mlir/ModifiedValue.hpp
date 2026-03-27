#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

namespace Bebra::MLIR {
enum class DataStoreType {
    NHWC,
    NCHW,
    UNDEFINED
};

struct ModifiedValue {
    mlir::Value value;
    DataStoreType dstype;
    mlir::Type type;
};

}
