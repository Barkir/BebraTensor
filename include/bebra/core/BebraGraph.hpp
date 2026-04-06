// BebraGraph.hpp

#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// #include "BebraAttr.hpp"
#include "BebraErr.hpp"
#include "BebraNode.hpp"
#include "BebraTensor.hpp"
#include "bebra/mlir/MLIRPrinter.hpp"

// #include "onnx_proto/onnx.proto3.pb.h"

namespace Bebra {
namespace Core {

const size_t DEFAULT_DATA_SIZE = 64;

struct BebraGraph {
    onnx::ModelProto model_;
    std::vector<BebraNode> nodes_;
    std::unordered_map<std::string, BebraTensor>
        tensor_map_; // this is bcs all tensor have different names in graph, so we can identify them by names
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::vector<std::string> initializers_;

public: // constructors
    BebraGraph(const std::string& modelPath) {
        std::ifstream file(modelPath, std::ios::binary);
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
    void convertOnnxToValueInfo(const onnx::GraphProto& graph);
    bool verifyGraph();
    void countOutputShapes();
    void dumpBebra(std::ofstream& stream);

public: // helper methods
    const BebraTensor& getTensor(const std::string& tensor_name) const {
        auto&& tensor = tensor_map_.find(tensor_name);
        if (tensor != tensor_map_.end()) {
            return tensor->second;
        }

        throw BebraErr("No such tensor " + tensor_name + " in tensor_map_...");
    }

    std::string convertToMlir(const std::string& file_name) {
        MSG("MLIR PRINTER\n");
        MSG("// \t\tINITIALIZERS\n");

        for (auto&& initializer : initializers_) {
            LOG("{} ", initializer);
        }
        MSG("\n\t\t // // //\n");
        MSG("// \t\tINPUTS\n");

        for (auto&& input : inputs_) {
            LOG("{} ", input);
        }

        MSG("\n\t\t // // //\n");
        MLIR::MLIRPrinter printer(*this);

        std::string result;
        printer.generate(*this, result);
        printer.dump(file_name, result);

        return result;
    }
};

} // namespace Core
} // namespace Bebra
