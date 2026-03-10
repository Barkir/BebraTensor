#pragma once
namespace Bebra::Ops {
// OpVariant for every node type
using OpVariant = std::variant<OpVoid,OpConv,OpGemm,OpAdd,OpRelu,OpMul,OpMatMul,OpMaxPool,OpReduceMean,OpReshape,OpSigmoid,OpFlatten>;
}
