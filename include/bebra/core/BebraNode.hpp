// BebraNode.hpp

#pragma once
#include <string>
#include <vector>
#include "bebra/core/BebraAttr.hpp"
// #include "bebra/core/BebraGraph.hpp"
#include "bebra/ops/BebraOperators.hpp"
#include "bebra/ops/BebraVariant.hpp"

namespace Bebra {
namespace Core {

struct BebraNode {
    Ops::OpVariant op_;

    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;

    BebraNode() = default;

};


} // end of Core :0
} // end of Bebra :0




