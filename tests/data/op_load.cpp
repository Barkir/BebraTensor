#include <gtest/gtest.h>
#include <filesystem>

#include "bebra/core/BebraGraph.hpp"
#include "bebra/ops/BebraOperators.hpp"
#include "helpers.hpp"

TEST(OpLoading, Conv) {
    const std::string model_path = get_model_path("third_party/resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    for (const auto& node : graph.nodes_) {
            std::string op_type;
        std::visit([&op_type](const auto& op) {
            op_type = op.getOpType();
        }, node.op_);

        if (op_type == "Conv") {

            Bebra::Ops::OpConv op;
            EXPECT_NO_THROW({
                auto k1 = op.kernel_shape;
            });

            EXPECT_NO_THROW({
                auto k2 = op.group;
            });


            EXPECT_NO_THROW({
                auto k3 = op.dilations;
            });

            EXPECT_NO_THROW({
                auto k4 = op.pads;
            });

            EXPECT_NO_THROW({
                auto k4 = op.strides;
            });
        }
    }
}



TEST(OpLoading, Gemm) {
    const std::string model_path = get_model_path("third_party/resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);

    for (const auto& node : graph.nodes_) {
            std::string op_type;
        std::visit([&op_type](const auto& op) {
            op_type = op.getOpType();
        }, node.op_);

        if (op_type == "Gemm") {
            ASSERT_NO_THROW(Bebra::Ops::OpGemm op);

            Bebra::Ops::OpGemm op;
            EXPECT_NO_THROW({
                auto k1 = op.alpha;
            });

            EXPECT_NO_THROW({
                auto k2 = op.beta;
            });


            EXPECT_NO_THROW({
                auto k3 = op.transA;
            });

            EXPECT_NO_THROW({
                auto k4 = op.transB;
            });
        }
    }
}


TEST(OpLoading, MaxPool) {
    auto model_path = get_model_path("third_party/resnet50-v1-7.onnx");
    Bebra::Core::BebraGraph graph(model_path);
    for (auto& node : graph.nodes_) {
            std::string op_type;
        std::visit([&op_type](const auto& op) {
            op_type = op.getOpType();
        }, node.op_);

        if (op_type == "MaxPool") {

        Bebra::Ops::OpMaxPool op;
        EXPECT_NO_THROW({
            auto k1 = op.kernel_shape;
        });

        EXPECT_NO_THROW({
            auto k2 = op.auto_pad;
        });

        EXPECT_NO_THROW({
            auto k3 = op.ceil_mode;
        });

        EXPECT_NO_THROW({
            auto k4 = op.dilations;
        });

        EXPECT_NO_THROW({
            auto k4 = op.pads;
        });

        EXPECT_NO_THROW({
            auto k4 = op.storage_order;
        });

        EXPECT_NO_THROW({
            auto k4 = op.strides;
        });
        }
    }
}


