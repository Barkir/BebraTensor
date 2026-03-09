#pragma once
namespace Bebra::Ops {
// OpVariant for every node type
using OpVariant = std::variant<OpConv,OpGemm,OpAdd,OpRelu,OpMul,OpMatMul,OpMaxPool,OpReduceMean,OpReshape,OpSigmoid,OpGlobalAveragePool,OpFlatten>;
}
