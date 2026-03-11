// BebraNode.hpp

#pragma once
#include "bebra/core/BebraAttr.hpp"
#include <string>
#include <vector>
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

} // namespace Core
} // namespace Bebra
