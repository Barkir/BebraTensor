#include <gtest/gtest.h>
#include <filesystem>

#include "bebra/core/BebraGraph.hpp"
#include "bebra/ops/BebraOperations.hpp"

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

static std::string get_model_path(std::string path) {
    return std::filesystem::path(TEST_DATA_DIR) / "third_party" / path;
}

TEST(OpLoading, Conv) {
    const std::string model_path = get_model_path("resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    for (const auto& node : graph.nodes_) {
        if (node.op_type_ == "Conv") {
            ASSERT_NO_THROW(Bebra::Ops::OpConv op(&node));

            Bebra::Ops::OpConv op(&node);
            EXPECT_NO_THROW({
                auto k1 = op.kernel_shape();
            });

            EXPECT_NO_THROW({
                auto k2 = op.group();
            });


            EXPECT_NO_THROW({
                auto k3 = op.dilations();
            });

            EXPECT_NO_THROW({
                auto k4 = op.pads();
            });

            EXPECT_NO_THROW({
                auto k4 = op.strides();
            });
        }
    }
}

TEST(OpNotLoading, ConvThrow) {
    const std::string model_path = get_model_path("resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    for (const auto& node : graph.nodes_) {
        if (node.op_type_ == "MatMul") {
            EXPECT_THROW({
                Bebra::Ops::OpConv op(&node);
                }, Bebra::Core::BebraErr);
        }
    }
}


