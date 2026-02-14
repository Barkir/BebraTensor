#include "bebra/core/BebraGraph.hpp"
#include "bebra/core/BebraTensor.hpp"

namespace Bebra {
namespace Core {

void BebraGraph::convertOnnxToBebraGraph(onnx::GraphProto& graph) {
    convertOnnxToBebraInitializer(graph);
    convertOnnxToBebraInput(graph);
    convertOnnxToBebraNode(graph);
}

void BebraGraph::convertOnnxToBebraInput(onnx::GraphProto& graph) {
    for (const auto& input : graph.input()) {
        BebraTensor t(input.name());
        tensor_map_.emplace(t.name_, std::move(t));
    }

    //TODO add dtype, bla-bla-bla
}

void BebraGraph::convertOnnxToBebraNode(onnx::GraphProto& graph) {
    for (const auto& onnx_node : graph.node()) {
        BebraNode node(onnx_node.op_type());

        for (const auto& input : onnx_node.input()) {
            node.inputs_.push_back(input);
        }

        for (const auto& output : onnx_node.output()) {
            node.outputs_.push_back(output);
        }

        for (const auto& attr : onnx_node.attribute()) {
            node.attrs_[attr.name()] = parseAttr(attr);
        }

        nodes_.push_back(std::move(node));
    }
}

void BebraGraph::convertOnnxToBebraInitializer(onnx::GraphProto& graph) {
    for (const auto& initializer : graph.initializer()) {
        BebraTensor t(initializer.name());
        tensor_map_.emplace(t.name_, std::move(t));

        //TODO add dtype
        //TODO add data
    }
}

} // end of Core :0
} // end of Bebra :0
