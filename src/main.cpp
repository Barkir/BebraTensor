
#include <fstream>
#include <iostream>

#include "onnx_proto/onnx.proto3.pb.h"
#include "bebra/core/BebraGraph.hpp"

const std::string some_model = "../third_party/model_quantized.onnx";

int main(void) {
    Bebra::Core::BebraGraph graph(some_model);
}
