#include <gtest/gtest.h>
#include <filesystem>
#include "bebra/core/BebraGraph.hpp"

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

    static std::string get_model_path(std::string path) {
        return std::filesystem::path(TEST_DATA_DIR) / "third_party" / path;
    }

TEST(GraphLoading, LoadsWithoutMissingTensors) {
    const std::string model_path = get_model_path("resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    std::cout << "\n  Loaded " << graph.nodes_.size() << " nodes, "
              << graph.tensor_map_.size() << " tensors" << std::endl;

    for (const auto& node : graph.nodes_) {
        for (const auto& input_name : node.inputs_) {
            auto it = graph.tensor_map_.find(input_name);
            ASSERT_NE(it, graph.tensor_map_.end())
                << "Node '" << node.op_type_ << "' missing input: " << input_name;
        }
    }
}

TEST(GraphLoading, HasExpectedStructure) {
    const std::string model_path = get_model_path("resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    EXPECT_FALSE(graph.nodes_.empty()) << "Graph has no nodes";
    EXPECT_FALSE(graph.tensor_map_.empty()) << "Graph has no tensors";

    // Check for transformer-specific ops
    bool has_gemm = false;
    bool has_relu = false;
    for (const auto& node : graph.nodes_) {
        if (node.op_type_ == "Gemm") has_gemm = true;
        if (node.op_type_ == "Relu") has_relu = true;
    }

    EXPECT_TRUE(has_gemm) << "Expected Gemm in transformer model";
    EXPECT_TRUE(has_relu) << "Expected Softmax in transformer model";
}


// Main provided by gtest_main
