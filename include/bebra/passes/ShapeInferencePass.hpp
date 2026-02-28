// analytical pass
// pass to get the size of all input, output tensors
// by traversing the graph

#pragma once
#include "bebra/core/BebraPass.hpp"

namespace Bebra {
namespace Pass {

class ShapeInferencePass : public BebraPass {
    public:
        std::string name() const override { return "shape-inference-pass"; }
        std::string desc() const override { return "analytical; get all tensor shaped thru traversing graph"; }
        std::vector<std::string> deps() const {return {};}
        bool run(Core::BebraGraph& graph) override;

    private: // codegen ???
        bool inferConv(Core::BebraNode& node);
        bool inferGemm(Core::BebraNode& node);
        bool inferAdd(Core::BebraNode& node);
        bool inferRelu(Core::BebraNode& node);
        bool inferMatMul(Core::BebraNode& node);
        bool inferMul(Core::BebraNode& node);
        bool inferMaxPool(Core::BebraNode& node);

    private: // fields
        Core::BebraGraph* graph_ = nullptr;

};

} // end of Pass :0
} // end of Bebra :0
