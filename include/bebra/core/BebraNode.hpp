// BebraNode.hpp

#pragma once
#include <string>
#include <vector>
#include "bebra/core/BebraAttr.hpp"

namespace Bebra {
namespace Core {

struct BebraNode {
    std::string                 op_type_;
    std::vector<std::string>    inputs_;
    std::vector<std::string>    outputs_;
    std::unordered_map<std::string, Attr> attrs_;

    BebraNode(std::string op_type) : op_type_(op_type) {
        std::cout << "Created node with type: " << op_type_ << std::endl;

        //TODO add inputs bla-bla-bla
    }

};

} // end of Core :0
} // end of Bebra :0




