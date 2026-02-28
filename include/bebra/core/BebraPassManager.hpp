#pragma once
#include <vector>
#include <memory>
#include <string>
#include "bebra/core/BebraGraph.hpp"
#include "bebra/core/BebraPass.hpp"

#include "bebra/passes/ShapeInferencePass.hpp"

namespace Bebra {
namespace Pass {

class BebraPassManager {
public:
    void registerPass(std::unique_ptr<BebraPass> pass);

    void run(Core::BebraGraph& graph);

    bool runPass(const std::string& name, Core::BebraGraph& graph);

    std::vector<std::string> getRegisteredPasses() const;

private:
    std::vector<std::unique_ptr<BebraPass>> passes_;
};

} // namespace Pass :0
} // namespace Bebra :0
