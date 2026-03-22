// BebraGraph.cpp

#include <fstream>
#include <functional>

#include "bebra/core/BebraGraph.hpp"
#include "bebra/core/BebraTensor.hpp"
#include "bebra/ops/BebraFactory.hpp"

namespace Bebra {
namespace Core {

void BebraGraph::convertOnnxToBebraGraph(const onnx::GraphProto& graph) {
    convertOnnxToBebraInitializer(graph);
    convertOnnxToBebraInput(graph);
    convertOnnxToBebraOutput(graph);
    convertOnnxToBebraNode(graph);

    // =====================================

    // countOutputShapes();
}

void BebraGraph::countOutputShapes() {
    for (auto&& node : nodes_) {
        std::visit(
            [this](auto& op) {
                LOG("counting output shape of op {} \n", op.getOpType());
                op.countOutputShape(*this);
            },

            node.op_);
    }
}

void BebraGraph::convertOnnxToBebraInput(const onnx::GraphProto& graph) {
    for (const auto& input : graph.input()) {
        BebraTensor t(input);
        tensor_map_.emplace(t.name_, std::move(t));
        inputs_.push_back(t.name_);
    }
}

void BebraGraph::convertOnnxToBebraOutput(const onnx::GraphProto& graph) {
    for (const auto& output : graph.output()) {
        BebraTensor t(output);
        tensor_map_.emplace(t.name_, std::move(t));
        outputs_.push_back(t.name_);
    }
}

void BebraGraph::convertOnnxToBebraNode(const onnx::GraphProto& graph) {
    for (const auto& onnx_node : graph.node()) {
        BebraNode node;

        for (const auto& input : onnx_node.input()) {
            if (tensor_map_.find(input) == tensor_map_.end()) {
                throw BebraErr("Input at node " + input + "not found.");
            }
            node.inputs_.push_back(input);
        }

        for (const auto& output : onnx_node.output()) {
            BebraTensor t(output);
            tensor_map_.emplace(output, std::move(t));

            node.outputs_.push_back(output);
        }

        std::unordered_map<std::string, Attr> attrs;
        if (onnx_node.attribute_size() != 0) {
            auto&& sz = onnx_node.attribute_size();
            for (int i = 0; i < sz; ++i) {
                auto&& attr = onnx_node.attribute(i);
                if (!attr.IsInitialized()) {
                    LOG("attr#{} not initialized!\n", i);
                    continue;
                }
                LOG("initialized attr with name {}\n", attr.name());
                attrs.emplace(attr.name(), Attr(parseAttr(attr)));
            }
        }

        const std::string& op_type = onnx_node.op_type();
        node.op_ = Ops::CreateOp(op_type, onnx_node, node.inputs_, node.outputs_, attrs);

        std::visit(
            [this](const auto& op) {
                if (!op.verify(*this)) {
                    throw BebraErr("Verification failed for op: " + std::string(typeid(op).name()));
                }
            },
            node.op_);

        nodes_.push_back(std::move(node));
    }
}

void BebraGraph::convertOnnxToBebraInitializer(const onnx::GraphProto& graph) {
    for (const auto& initializer : graph.initializer()) {
        BebraTensor t(initializer);
        tensor_map_.emplace(t.name_, std::move(t));
        initializers_.push_back(t.name_);
    }
}

void BebraGraph::dumpBebra(std::ofstream& stream) {
    stream << "digraph{" << "\n";
    for (const auto& node : nodes_) {
        std::string op_type;
        std::vector<std::string> attrs;
        std::visit(
            [&op_type, &attrs](const auto& op) {
                op_type = op.getOpType();
                attrs = op.getAttrsString();
            },
            node.op_);

        stream << "node" << &node << "[\"label\"=\"{" << op_type;
        for (const auto& attr : attrs) {
            stream << "|" << attr << "\n";
        }
        stream << "}\",shape=Mrecord style=filled, fillcolor=\"" << getNodeColor(op_type) << "\"]" << "\n";

        for (const auto& input : node.inputs_) {
            stream << "tensor" << input << "[label=" << input << ", shape=cylinder, style=filled, fillcolor=\""
                   << BEBRA_TENSOR << "\"]" << "\n";
            stream << "tensor" << input << "->" << "node" << &node << "\n";
        }

        for (const auto& output : node.outputs_) {
            stream << "tensor" << output << "[label=" << output << ", shape=cylinder, style=filled, fillcolor=\""
                   << BEBRA_TENSOR << "\"]" << "\n";
            stream << "node" << &node << "->" << "tensor" << output << "\n";
        }
    }
    stream << "}" << "\n";
}

} // namespace Core
} // namespace Bebra
