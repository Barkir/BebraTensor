
#include <filesystem>
#include <fstream>
#include <iostream>

#include "bebra/core/BebraConst.hpp"
#include "bebra/core/BebraGraph.hpp"
#include "bebra/core/BebraPassManager.hpp"
#include "bebra/ops/BebraOperators.hpp"
#include "onnx_proto/onnx.proto3.pb.h"

#ifndef PROJECT_ROOT
#define PROJECT_ROOT "."
#endif

static std::string get_dot_path(const std::string& filename) {
    auto path = std::filesystem::path(PROJECT_ROOT) / "dot" / filename;
    return std::filesystem::absolute(path).string();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << BEBRA_DIM BEBRA_ITALIC "Please enter model to parse..." RESET_DIM << std::endl;
        std::cerr << FG_TURQUOISE BEBRA_BOLD "hint: enter `--dump` flag to get your graph image" BEBRA_RESET
                  << std::endl;
        return 0;
    }

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dump")) {
            if (i != argc - 1) {
                std::filesystem::path model_path(argv[i + 1]);
                std::string dot_name = model_path.stem().string() + ".dot";

                Bebra::Core::BebraGraph graph(argv[i + 1]);
                std::string dot_path = get_dot_path(dot_name);
                std::ofstream stream;
                stream.open(dot_path);
                if (!stream.is_open()) {
                    throw Bebra::Core::BebraErr("Stream " + dot_path + " not opened!");
                }

                graph.dumpBebra(stream);
                return 0;
            } else {
                std::cerr << BEBRA_DIM BEBRA_ITALIC "No model entered to parse after --dump flag!" RESET_DIM
                          << std::endl;
                return 0;
            }
        }

        if (strcmp(argv[i], "--dump")) {
            Bebra::Core::BebraGraph graph(argv[i]);
            graph.convertToMlir();
            // VerifyGraph

            Bebra::Pass::BebraPassManager pm;

            // need to add plugin linkage
            // pm.registerPass(std::make_unique<Bebra::Pass::ShapeInferencePass>());
            pm.run(graph);
            // ... do smth here
            return 0;
        }
    }
}
