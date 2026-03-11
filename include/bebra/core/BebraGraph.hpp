// BebraGraph.hpp

#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <iostream>

// #include "BebraAttr.hpp"
#include "bebra/mlir/MLIRPrinter.hpp"
#include "BebraTensor.hpp"
#include "BebraNode.hpp"
#include "BebraErr.hpp"

// #include "onnx_proto/onnx.proto3.pb.h"

namespace Bebra {
namespace Core {

struct BebraGraph {

    onnx::ModelProto model_;
    std::vector<BebraNode> nodes_;
    std::unordered_map<std::string, BebraTensor> tensor_map_; // this is bcs all tensor have different names in graph, so we can identify them by names
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;

    public: // constructors
    BebraGraph(const std::string& modelPath) {
        std::ifstream file (modelPath, std::ios::binary);
        if (!file) {
            throw BebraErr("Cannot open file: " + modelPath);
        }

        if (!model_.ParseFromIstream(&file)) {
            throw BebraErr("Cannot parse model from file: " + modelPath);
        }

        // then converting to own graph
        convertOnnxToBebraGraph(model_.graph());
    }

    public: // methods
        void convertOnnxToBebraGraph(const onnx::GraphProto& graph);
        void convertOnnxToBebraInput(const onnx::GraphProto& graph);
        void convertOnnxToBebraInitializer(const onnx::GraphProto& graph);
        void convertOnnxToBebraNode(const onnx::GraphProto& graph);
        void convertOnnxToBebraOutput(const onnx::GraphProto& graph);
        bool verifyGraph();
        void dumpBebra(std::ofstream& stream);

    public: // helper methods
        const BebraTensor& getTensor(const std::string& tensor_name) const {
            auto&& tensor = tensor_map_.find(tensor_name);
            if (tensor != tensor_map_.end()) {
                return tensor->second;
            }

            throw BebraErr("No such tensor " + tensor_name + " in tensor_map_...");
        }

        void convertToMlir() {
            std::cout << "MLIR PRINTER" << std::endl;
            MLIR::MLIRPrinter printer(*this);
            printer.generate(*this);
        }


};

} // end of Core :0
} // end of Bebra :0
