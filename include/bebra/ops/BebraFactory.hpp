#pragma once
#include "bebra/core/BebraNode.hpp"
#include "bebra/core/BebraAttr.hpp"
#include "bebra/ops/BebraOperators.hpp"

namespace Bebra::Ops {

OpVariant CreateOp(const std::string& op_type,
                   const onnx::NodeProto& onnx_node,
                   const std::vector<std::string>& inputs,
                   const std::vector<std::string>& outputs,
                   const std::unordered_map<std::string, Core::Attr>& attrs);
OpVariant CreateOpConv (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
OpVariant CreateOpGemm (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
OpVariant CreateOpAdd (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
OpVariant CreateOpRelu (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
OpVariant CreateOpMul (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
OpVariant CreateOpMatMul (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
OpVariant CreateOpMaxPool (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
OpVariant CreateOpReduceMean (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
OpVariant CreateOpReshape (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
OpVariant CreateOpSigmoid (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
OpVariant CreateOpGlobalAveragePool (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
OpVariant CreateOpFlatten (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
}
