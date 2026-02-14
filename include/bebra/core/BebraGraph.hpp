// BebraGraph.hpp

#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>

#include "BebraNode.hpp"
#include "BebraTensor.hpp"
#include "BebraAttr.hpp"
#include "BebraErr.hpp"

#include "onnx_proto/onnx.proto3.pb.h"

namespace Bebra {
namespace Core {

struct BebraGraph {

    private:
    std::vector<BebraNode> nodes_;
    std::unordered_map<std::string, BebraTensor> tensor_map_; // this is bcs all tensor have different names in graph, so we can identify them by names

    public: // constructors
    BebraGraph(const std::string& modelPath) {
        std::ifstream file (modelPath, std::ios::binary);
        if (!file) {
            throw BebraErr("Cannot open file: " + modelPath);
        }

        onnx::ModelProto model;
        if (!model.ParseFromIstream(&file)) {
            throw BebraErr("Cannot parse model from file: " + modelPath);
        }

        // then converting to own graph
    }

    public: // methods
        void convertOnnxToBebraGraph(onnx::GraphProto& graph);
        void convertOnnxToBebraInput(onnx::GraphProto& graph);
        void convertOnnxToBebraInitializer(onnx::GraphProto& graph);
        void convertOnnxToBebraNode(onnx::GraphProto& graph);


};

} // end of Core :0
} // end of Bebra :0
