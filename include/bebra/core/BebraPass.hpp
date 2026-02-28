#pragma once
#include "bebra/core/BebraGraph.hpp"
#include <string>

namespace Bebra {
namespace Pass {

class BebraPass {
public:
    virtual ~BebraPass() = default;


    virtual std::string name() const = 0;
    virtual std::string desc() const = 0;

    virtual bool run(Core::BebraGraph& graph) = 0;
    virtual std::vector<std::string> deps() const = 0; // passes needed to run this pass

};

using BebraPassCreator = std::unique_ptr<BebraPass>(*)();

} // end of Pass :0
} // end of Bebra :0
