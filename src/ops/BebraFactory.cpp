#include "bebra/ops/BebraFactory.hpp"
namespace Bebra::Ops {

OpVariant CreateOp(const std::string& op_type,
                   const onnx::NodeProto& onnx_node,
                   const std::vector<std::string>& inputs,
                   const std::vector<std::string>& outputs,
                   const std::unordered_map<std::string, Core::Attr>& attrs)

{
        if (op_type == "Conv") {
        return CreateOpConv(onnx_node, inputs, outputs, attrs);
    }
    if (op_type == "Gemm") {
        return CreateOpGemm(onnx_node, inputs, outputs, attrs);
    }
    if (op_type == "Add") {
        return CreateOpAdd(onnx_node, inputs, outputs, attrs);
    }
    if (op_type == "Relu") {
        return CreateOpRelu(onnx_node, inputs, outputs, attrs);
    }
    if (op_type == "Mul") {
        return CreateOpMul(onnx_node, inputs, outputs, attrs);
    }
    if (op_type == "MatMul") {
        return CreateOpMatMul(onnx_node, inputs, outputs, attrs);
    }
    if (op_type == "MaxPool") {
        return CreateOpMaxPool(onnx_node, inputs, outputs, attrs);
    }
    if (op_type == "ReduceMean") {
        return CreateOpReduceMean(onnx_node, inputs, outputs, attrs);
    }
    if (op_type == "Reshape") {
        return CreateOpReshape(onnx_node, inputs, outputs, attrs);
    }
    if (op_type == "Sigmoid") {
        return CreateOpSigmoid(onnx_node, inputs, outputs, attrs);
    }
    if (op_type == "GlobalAveragePool") {
        return CreateOpGlobalAveragePool(onnx_node, inputs, outputs, attrs);
    }
    if (op_type == "Flatten") {
        return CreateOpFlatten(onnx_node, inputs, outputs, attrs);
    }

}
OpVariant CreateOpConv (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    OpConv op;
        if (inputs.size() < 2) {
            throw Core::BebraErr("Conv: expected 2 required input(s)");
        }
        op.input = inputs[0];
        op.weight = inputs[1];
        if (inputs.size() > 2 && !inputs[2].empty()) op.bias = inputs[2];

        if (outputs.size() < 1) {
            throw Core::BebraErr("Conv: expected 1 required output(s)");
        }
        op.output = outputs[0];

        {
            auto it = attrs.find("kernel_shape");
            if (it == attrs.end()) throw Core::BebraErr("Conv: missing required attr 'kernel_shape'");
            op.kernel_shape = it->second.getValRef<std::vector<int64_t>>();
        }
        {
            auto it = attrs.find("group");
            op.group = (it != attrs.end())
                ? it->second.getValRef<int64_t>()
                : int64_t(1);
        }
        {
            auto it = attrs.find("dilations");
            op.dilations = (it != attrs.end())
                ? it->second.getValRef<std::vector<int64_t>>()
                : std::vector<int64_t>({1, 1});
        }
        {
            auto it = attrs.find("pads");
            op.pads = (it != attrs.end())
                ? it->second.getValRef<std::vector<int64_t>>()
                : std::vector<int64_t>({0, 0});
        }
        {
            auto it = attrs.find("strides");
            op.strides = (it != attrs.end())
                ? it->second.getValRef<std::vector<int64_t>>()
                : std::vector<int64_t>({1, 1});
        }
	return op;
}
OpVariant CreateOpGemm (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    OpGemm op;
        if (inputs.size() < 2) {
            throw Core::BebraErr("Gemm: expected 2 required input(s)");
        }
        op.input_a = inputs[0];
        op.input_b = inputs[1];
        if (inputs.size() > 2 && !inputs[2].empty()) op.bias = inputs[2];

        if (outputs.size() < 1) {
            throw Core::BebraErr("Gemm: expected 1 required output(s)");
        }
        op.output = outputs[0];

        {
            auto it = attrs.find("alpha");
            op.alpha = (it != attrs.end())
                ? it->second.getValRef<float>()
                : float(1.0);
        }
        {
            auto it = attrs.find("beta");
            op.beta = (it != attrs.end())
                ? it->second.getValRef<float>()
                : float(1.0);
        }
        {
            auto it = attrs.find("transA");
            op.transA = (it != attrs.end())
                ? it->second.getValRef<int64_t>()
                : int64_t(0);
        }
        {
            auto it = attrs.find("transB");
            op.transB = (it != attrs.end())
                ? it->second.getValRef<int64_t>()
                : int64_t(0);
        }
	return op;
}
OpVariant CreateOpAdd (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    OpAdd op;
        if (inputs.size() < 2) {
            throw Core::BebraErr("Add: expected 2 required input(s)");
        }
        op.input_1 = inputs[0];
        op.input_2 = inputs[1];

        if (outputs.size() < 1) {
            throw Core::BebraErr("Add: expected 1 required output(s)");
        }
        op.output = outputs[0];

	return op;
}
OpVariant CreateOpRelu (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    OpRelu op;
        if (inputs.size() < 1) {
            throw Core::BebraErr("Relu: expected 1 required input(s)");
        }
        op.input = inputs[0];

        if (outputs.size() < 1) {
            throw Core::BebraErr("Relu: expected 1 required output(s)");
        }
        op.output = outputs[0];

	return op;
}
OpVariant CreateOpMul (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    OpMul op;
        if (inputs.size() < 2) {
            throw Core::BebraErr("Mul: expected 2 required input(s)");
        }
        op.input_1 = inputs[0];
        op.input_2 = inputs[1];

        if (outputs.size() < 1) {
            throw Core::BebraErr("Mul: expected 1 required output(s)");
        }
        op.output = outputs[0];

	return op;
}
OpVariant CreateOpMatMul (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    OpMatMul op;
        if (inputs.size() < 2) {
            throw Core::BebraErr("MatMul: expected 2 required input(s)");
        }
        op.input_a = inputs[0];
        op.input_b = inputs[1];

        if (outputs.size() < 1) {
            throw Core::BebraErr("MatMul: expected 1 required output(s)");
        }
        op.output = outputs[0];

	return op;
}
OpVariant CreateOpMaxPool (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    OpMaxPool op;
        if (inputs.size() < 1) {
            throw Core::BebraErr("MaxPool: expected 1 required input(s)");
        }
        op.input = inputs[0];

        if (outputs.size() < 1) {
            throw Core::BebraErr("MaxPool: expected 1 required output(s)");
        }
        op.output = outputs[0];
        if (outputs.size() > 1) op.indices = outputs[1];

        {
            auto it = attrs.find("kernel_shape");
            if (it == attrs.end()) throw Core::BebraErr("MaxPool: missing required attr 'kernel_shape'");
            op.kernel_shape = it->second.getValRef<std::vector<int64_t>>();
        }
        {
            auto it = attrs.find("auto_pad");
            op.auto_pad = (it != attrs.end())
                ? it->second.getValRef<std::string>()
                : std::string("NOTSET");
        }
        {
            auto it = attrs.find("ceil_mode");
            op.ceil_mode = (it != attrs.end())
                ? it->second.getValRef<int64_t>()
                : int64_t(0);
        }
        {
            auto it = attrs.find("dilations");
            op.dilations = (it != attrs.end())
                ? it->second.getValRef<std::vector<int64_t>>()
                : std::vector<int64_t>({1, 1});
        }
        {
            auto it = attrs.find("pads");
            op.pads = (it != attrs.end())
                ? it->second.getValRef<std::vector<int64_t>>()
                : std::vector<int64_t>({0, 0});
        }
        {
            auto it = attrs.find("storage_order");
            op.storage_order = (it != attrs.end())
                ? it->second.getValRef<int64_t>()
                : int64_t(0);
        }
        {
            auto it = attrs.find("strides");
            op.strides = (it != attrs.end())
                ? it->second.getValRef<std::vector<int64_t>>()
                : std::vector<int64_t>({1, 1});
        }
	return op;
}
OpVariant CreateOpReduceMean (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    OpReduceMean op;
        if (inputs.size() < 1) {
            throw Core::BebraErr("ReduceMean: expected 1 required input(s)");
        }
        op.input = inputs[0];
        if (inputs.size() > 1 && !inputs[1].empty()) op.axes_t = inputs[1];

        if (outputs.size() < 1) {
            throw Core::BebraErr("ReduceMean: expected 1 required output(s)");
        }
        op.output = outputs[0];

        {
            auto it = attrs.find("axes");
            if (it == attrs.end()) throw Core::BebraErr("ReduceMean: missing required attr 'axes'");
            op.axes = it->second.getValRef<std::vector<int64_t>>();
        }
        {
            auto it = attrs.find("keepdims");
            op.keepdims = (it != attrs.end())
                ? it->second.getValRef<int64_t>()
                : int64_t(1);
        }
	return op;
}
OpVariant CreateOpReshape (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    OpReshape op;
        if (inputs.size() < 2) {
            throw Core::BebraErr("Reshape: expected 2 required input(s)");
        }
        op.input = inputs[0];
        op.shape_t = inputs[1];

        if (outputs.size() < 1) {
            throw Core::BebraErr("Reshape: expected 1 required output(s)");
        }
        op.output = outputs[0];

        {
            auto it = attrs.find("shape");
            if (it == attrs.end()) throw Core::BebraErr("Reshape: missing required attr 'shape'");
            op.shape = it->second.getValRef<std::vector<int64_t>>();
        }
	return op;
}
OpVariant CreateOpSigmoid (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    OpSigmoid op;
        if (inputs.size() < 1) {
            throw Core::BebraErr("Sigmoid: expected 1 required input(s)");
        }
        op.input = inputs[0];

        if (outputs.size() < 1) {
            throw Core::BebraErr("Sigmoid: expected 1 required output(s)");
        }
        op.output = outputs[0];

	return op;
}
OpVariant CreateOpGlobalAveragePool (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    OpGlobalAveragePool op;
        if (inputs.size() < 1) {
            throw Core::BebraErr("GlobalAveragePool: expected 1 required input(s)");
        }
        op.input = inputs[0];

        if (outputs.size() < 1) {
            throw Core::BebraErr("GlobalAveragePool: expected 1 required output(s)");
        }
        op.output = outputs[0];

	return op;
}
OpVariant CreateOpFlatten (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    OpFlatten op;
        if (inputs.size() < 1) {
            throw Core::BebraErr("Flatten: expected 1 required input(s)");
        }
        op.input = inputs[0];

        if (outputs.size() < 1) {
            throw Core::BebraErr("Flatten: expected 1 required output(s)");
        }
        op.output = outputs[0];

        {
            auto it = attrs.find("axis");
            op.axis = (it != attrs.end())
                ? it->second.getValRef<int64_t>()
                : int64_t(1);
        }
	return op;
}
}
