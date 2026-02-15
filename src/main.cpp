
#include <fstream>
#include <iostream>

#include "onnx_proto/onnx.proto3.pb.h"
#include "bebra/core/BebraGraph.hpp"
#include "bebra/ops/BebraOperations.hpp"

const std::string some_model = "../third_party/model_quantized.onnx";
const std::string other_model = "../third_party/resnet50-v1-7.onnx";

int main(void) {
    Bebra::Core::BebraGraph graph(other_model);

    for (const auto& node : graph.nodes_) {
        if (node.op_type_ == "Conv") {
            Bebra::Ops::OpConv op(&node);

        }
    }

}
