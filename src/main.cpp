
#include <fstream>
#include <iostream>

#include "onnx_proto/onnx.proto3.pb.h"
#include "bebra/core/BebraGraph.hpp"

const std::string some_model = "../third_party/model_quantized.onnx";

int main(void) {
    Bebra::Core::BebraGraph graph(some_model);

    std::cout << "Got " << graph.nodes_.size() << " nodes." << std::endl;

    for (const auto& n : graph.nodes_) {
        for (const auto& t : n.inputs_) {
            auto it = graph.tensor_map_.find(t);
            if (it == graph.tensor_map_.end()) {
                throw Bebra::Core::BebraErr("Tensor with name " + t + " not found.");
            }
            std::cout << BG_BRIGHT_GREEN << t << RESET << std::endl;
        }

    }
}
