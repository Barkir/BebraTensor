#include <gtest/gtest.h>
#include <filesystem>

#include "bebra/core/BebraGraph.hpp"
#include "bebra/ops/BebraOperators.hpp"
#include "helpers.hpp"

#include <gtest/gtest.h>
#include <filesystem>


TEST(add0d, VerifyIR) {
    auto file = get_model_path("models/01_arithmetic.onnx");
    auto check = get_checker_path(file);
    auto ll_path = get_ll_path(file);

    Bebra::Core::BebraGraph graph(file);
    auto res = graph.convertToMlir(ll_path);


    VerifyIR(res, check);

}
