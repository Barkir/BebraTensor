// BebraGraph.cpp

#include "bebra/core/BebraGraph.hpp"
#include "bebra/core/BebraTensor.hpp"

namespace Bebra {
namespace Core {

void BebraGraph::convertOnnxToBebraGraph(const onnx::GraphProto& graph) {
    convertOnnxToBebraInitializer(graph);
    convertOnnxToBebraInput(graph);
    convertOnnxToBebraOutput(graph);
    convertOnnxToBebraNode(graph);
}

void BebraGraph::convertOnnxToBebraInput(const onnx::GraphProto& graph) {
    for (const auto& input : graph.input()) {
        BebraTensor t(input.name());
        tensor_map_.emplace(t.name_, std::move(t));
    }

    //TODO add dtype, bla-bla-bla
}

void BebraGraph::convertOnnxToBebraOutput(const onnx::GraphProto& graph) {
    for (const auto& output : graph.output()) {
        BebraTensor t(output.name());
        tensor_map_.emplace(t.name_, std::move(t));
    }
}

void BebraGraph::convertOnnxToBebraNode(const onnx::GraphProto& graph) {
    for (const auto& onnx_node : graph.node()) {
        BebraNode node(onnx_node.op_type());

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

        for (const auto& attr : onnx_node.attribute()) {
            node.attrs_[attr.name()] = parseAttr(attr);
        }

        nodes_.push_back(std::move(node));
    }
}

void BebraGraph::convertOnnxToBebraInitializer(const onnx::GraphProto& graph) {
    for (const auto& initializer : graph.initializer()) {
        BebraTensor t(initializer.name());
        tensor_map_.emplace(t.name_, std::move(t));

        //TODO add dtype
        //TODO add data
    }
}

} // end of Core :0
} // end of Bebra :0
