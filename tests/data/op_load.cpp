#include <gtest/gtest.h>
#include <filesystem>

#include "bebra/core/BebraGraph.hpp"
#include "bebra/ops/BebraOperators.hpp"

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

TEST(OpLoading, Gemm) {
    const std::string model_path = get_model_path("resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    for (const auto& node : graph.nodes_) {
        if (node.op_type_ == "Gemm") {
            ASSERT_NO_THROW(Bebra::Ops::OpGemm op(&node));

            Bebra::Ops::OpGemm op(&node);
            EXPECT_NO_THROW({
                auto k1 = op.alpha();
            });

            EXPECT_NO_THROW({
                auto k2 = op.beta();
            });


            EXPECT_NO_THROW({
                auto k3 = op.transA();
            });

            EXPECT_NO_THROW({
                auto k4 = op.transB();
            });
        }
    }
}

TEST(OpNotLoading, GemmThrow) {
    const std::string model_path = get_model_path("resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    for (const auto& node : graph.nodes_) {
        if (node.op_type_ == "Conv") {
            EXPECT_THROW({
                Bebra::Ops::OpGemm op(&node);
                }, Bebra::Core::BebraErr);
        }
    }
}

TEST(OpLoading, MatMul) {
    auto model_path = get_model_path("resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);
    for (auto& node : graph.nodes_) {
        if (node.op_type_ == "MatMul") {
            ASSERT_NO_THROW(Bebra::Ops::OpMatMul op(&node));
        }
    }
}


TEST(OpLoading, Relu) {
    auto model_path = get_model_path("resnet50-v1-7.onnx");

    Bebra::Core::BebraGraph graph(model_path);
    for (auto& node : graph.nodes_) {
        if (node.op_type_ == "Relu") {
            ASSERT_NO_THROW(Bebra::Ops::OpRelu op(&node));
        }
    }
}


TEST(OpLoading, Add) {
    auto model_path = get_model_path("resnet50-v1-7.onnx");

    Bebra::Core::BebraGraph graph(model_path);
    for (auto& node : graph.nodes_) {
        if (node.op_type_ == "Add") {
            ASSERT_NO_THROW(Bebra::Ops::OpAdd op(&node));
        }
    }
}


TEST(OpLoading, Mul) {
    auto model_path = get_model_path("resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);
    for (auto& node : graph.nodes_) {
        if (node.op_type_ == "Mul") {
            ASSERT_NO_THROW(Bebra::Ops::OpMul op(&node));
        }
    }
}


TEST(OpLoading, MaxPool) {
    auto model_path = get_model_path("resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);
    for (auto& node : graph.nodes_) {
        if (node.op_type_ == "MaxPool") {
            ASSERT_NO_THROW(Bebra::Ops::OpMaxPool op(&node));

        Bebra::Ops::OpMaxPool op(&node);
        EXPECT_NO_THROW({
            auto k1 = op.kernel_shape();
        });

        EXPECT_NO_THROW({
            auto k2 = op.auto_pad();
        });

        EXPECT_NO_THROW({
            auto k3 = op.ceil_mode();
        });

        EXPECT_NO_THROW({
            auto k4 = op.dilations();
        });

        EXPECT_NO_THROW({
            auto k4 = op.pads();
        });

        EXPECT_NO_THROW({
            auto k4 = op.storage_order();
        });

        EXPECT_NO_THROW({
            auto k4 = op.strides();
        });
        }
    }
}


