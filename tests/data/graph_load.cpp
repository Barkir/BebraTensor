#include <gtest/gtest.h>
#include <filesystem>
#include "bebra/core/BebraGraph.hpp"

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

    static std::string get_model_path(std::string path) {
        return std::filesystem::path(TEST_DATA_DIR) / "tests" / path;
    }

TEST(GraphLoading1, LoadsWithoutMissingTensors) {
    const std::string model_path = get_model_path("third_party/resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    LOG("\n  Loaded {} nodes, {} tensors\n", graph.nodes_.size(), graph.tensor_map_.size());

    for (const auto& node : graph.nodes_) {
        std::string op_type;
        std::visit([&op_type](const auto& op) {
            op_type = op.getOpType();
        }, node.op_);

        for (const auto& input_name : node.inputs_) {
            auto it = graph.tensor_map_.find(input_name);
            ASSERT_NE(it, graph.tensor_map_.end())
                << "Node '" << op_type << "' missing input: " << input_name;
        }
    }
}

TEST(GraphLoading2, LoadsWithoutMissingTensors2) {
    const std::string model_path = get_model_path("third_party/mnist-8.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    std::cout << "\n  Loaded " << graph.nodes_.size() << " nodes, "
              << graph.tensor_map_.size() << " tensors" << "\n";

    for (const auto& node : graph.nodes_) {
        std::string op_type;
        std::visit([&op_type](const auto& op) {
            op_type = op.getOpType();
        }, node.op_);

        for (const auto& input_name : node.inputs_) {
            auto it = graph.tensor_map_.find(input_name);
            ASSERT_NE(it, graph.tensor_map_.end())
                << "Node '" << op_type << "' missing input: " << input_name;
        }
    }
}

TEST(GraphLoading3, LoadsWithoutMissingTensors3) {
    const std::string model_path = get_model_path("third_party/mnist-8.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    std::cout << "\n  Loaded " << graph.nodes_.size() << " nodes, "
              << graph.tensor_map_.size() << " tensors" << "\n";

    for (const auto& node : graph.nodes_) {
        std::string op_type;
        std::visit([&op_type](const auto& op) {
            op_type = op.getOpType();
        }, node.op_);

        for (const auto& input_name : node.inputs_) {
            auto it = graph.tensor_map_.find(input_name);
            ASSERT_NE(it, graph.tensor_map_.end())
                << "Node '" << op_type << "' missing input: " << input_name;
        }
    }
}

TEST(GraphLoading4, LoadsWithoutMissingTensors4) {
    const std::string model_path = get_model_path("tiny_onnx/01_arithmetic.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    std::cout << "\n  Loaded " << graph.nodes_.size() << " nodes, "
              << graph.tensor_map_.size() << " tensors" << "\n";

    for (const auto& node : graph.nodes_) {
        std::string op_type;
        std::visit([&op_type](const auto& op) {
            op_type = op.getOpType();
        }, node.op_);

        for (const auto& input_name : node.inputs_) {
            auto it = graph.tensor_map_.find(input_name);
            ASSERT_NE(it, graph.tensor_map_.end())
                << "Node '" << op_type << "' missing input: " << input_name;
        }
    }
}

// TEST(GraphLoading5, LoadsWithoutMissingTensors5) {
//     const std::string model_path = get_model_path("tiny_onnx/01_conv_bn_relu.onnx");
//     Bebra::Core::BebraGraph graph(model_path);
//
//     std::cout << "\n  Loaded " << graph.nodes_.size() << " nodes, "
//               << graph.tensor_map_.size() << " tensors" << "\n";
//
//     for (const auto& node : graph.nodes_) {
//         std::string op_type;
//         std::visit([&op_type](const auto& op) {
//             op_type = op.getOpType();
//         }, node.op_);
//
//         for (const auto& input_name : node.inputs_) {
//             auto it = graph.tensor_map_.find(input_name);
//             ASSERT_NE(it, graph.tensor_map_.end())
//                 << "Node '" << op_type << "' missing input: " << input_name;
//         }
//     }
// }

TEST(GraphLoading6, LoadsWithoutMissingTensors6) {
    const std::string model_path = get_model_path("tiny_onnx/02_gemm_classifier.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    std::cout << "\n  Loaded " << graph.nodes_.size() << " nodes, "
              << graph.tensor_map_.size() << " tensors" << "\n";

    for (const auto& node : graph.nodes_) {
        std::string op_type;
        std::visit([&op_type](const auto& op) {
            op_type = op.getOpType();
        }, node.op_);

        for (const auto& input_name : node.inputs_) {
            auto it = graph.tensor_map_.find(input_name);
            ASSERT_NE(it, graph.tensor_map_.end())
                << "Node '" << op_type << "' missing input: " << input_name;
        }
    }
}

TEST(GraphLoading, HasExpectedStructure) {
    const std::string model_path = get_model_path("third_party/resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    EXPECT_FALSE(graph.nodes_.empty()) << "Graph has no nodes";
    EXPECT_FALSE(graph.tensor_map_.empty()) << "Graph has no tensors";


    // Check for transformer-specific ops
    bool has_gemm = false;
    bool has_relu = false;
    for (const auto& node : graph.nodes_) {
        std::string op_type;
        std::visit([&op_type](const auto& op) {
            op_type = op.getOpType();
        }, node.op_);

        if (op_type == "Gemm") has_gemm = true;
        if (op_type == "Relu") has_relu = true;
    }

    EXPECT_TRUE(has_gemm) << "Expected Gemm in transformer model";
    EXPECT_TRUE(has_relu) << "Expected Softmax in transformer model";
}


// Main provided by gtest_main
