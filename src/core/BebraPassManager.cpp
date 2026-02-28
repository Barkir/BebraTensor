#include "bebra/core/BebraPassManager.hpp"
#include <iostream>

namespace Bebra {
namespace Pass {

void BebraPassManager::registerPass(std::unique_ptr<BebraPass> pass) {
    std::cout << "[PassManager] Registered: " << pass->name() << "\n";
    passes_.push_back(std::move(pass));
}

void BebraPassManager::run(Core::BebraGraph& graph) {
    std::cout << "[PassManager] Running " << passes_.size() << " passes\n";

    for (auto& pass : passes_) {
        std::cout << "[PassManager] Running: " << pass->name() << "\n";
        bool changed = pass->run(graph);
        std::cout << "[PassManager]   " << (changed ? "changed" : "no change") << "\n";
    }
}

} // namespace Pass :0
} // namespace Bebra :0
