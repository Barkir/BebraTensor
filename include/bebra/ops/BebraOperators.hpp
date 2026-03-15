#pragma once
#include <string>
namespace Bebra { namespace Core { class BebraGraph; } }
#include <vector>
#include <iostream>
#include "bebra/core/BebraErr.hpp"
#include "bebra/core/BebraColors.hpp"
#include "bebra/ops/BebraVisitor.hpp"
#include "bebra/ops/CountShapeHelpers.hpp"
namespace Bebra {
namespace Ops {
struct OpVoid {
void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
static constexpr const char* getOpType() { return "Void"; }
std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
const std::vector<std::string> getAttrsString() const {
    return {
        
    };
}




bool verify(const Core::BebraGraph& graph) const;
    };
struct OpConv {
void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
static constexpr const char* getOpType() { return "Conv"; }
std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
const std::vector<std::string> getAttrsString() const {
    return {
        "kernel_shape",
"group",
"dilations",
"pads",
"strides",
"auto_pad"
    };
}

        std::string input;
        std::string weight;
        std::string bias;

        std::string output;

    std::vector<int64_t> kernel_shape;
    int64_t group = int64_t(1);
    std::vector<int64_t> dilations = std::vector<int64_t>({1, 1});
    std::vector<int64_t> pads = std::vector<int64_t>({0, 0});
    std::vector<int64_t> strides = std::vector<int64_t>({1, 1});
    std::string auto_pad = std::string("NOTSET");

bool verify(const Core::BebraGraph& graph) const;
    };
struct OpGemm {
void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
static constexpr const char* getOpType() { return "Gemm"; }
std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
const std::vector<std::string> getAttrsString() const {
    return {
        "alpha",
"beta",
"transA",
"transB"
    };
}

        std::string input_a;
        std::string input_b;
        std::string bias;

        std::string output;

    float alpha = float(1.0f);
    float beta = float(1.0f);
    int64_t transA = int64_t(0);
    int64_t transB = int64_t(0);

bool verify(const Core::BebraGraph& graph) const;
    };
struct OpAdd {
void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
static constexpr const char* getOpType() { return "Add"; }
std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
const std::vector<std::string> getAttrsString() const {
    return {
        
    };
}

        std::string input_1;
        std::string input_2;

        std::string output;


bool verify(const Core::BebraGraph& graph) const;
    };
struct OpRelu {
void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
static constexpr const char* getOpType() { return "Relu"; }
std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
const std::vector<std::string> getAttrsString() const {
    return {
        
    };
}

        std::string input;

        std::string output;


bool verify(const Core::BebraGraph& graph) const;
    };
struct OpMul {
void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
static constexpr const char* getOpType() { return "Mul"; }
std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
const std::vector<std::string> getAttrsString() const {
    return {
        
    };
}

        std::string input_1;
        std::string input_2;

        std::string output;


bool verify(const Core::BebraGraph& graph) const;
    };
struct OpMatMul {
void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
static constexpr const char* getOpType() { return "MatMul"; }
std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
const std::vector<std::string> getAttrsString() const {
    return {
        
    };
}

        std::string input_a;
        std::string input_b;

        std::string output;


bool verify(const Core::BebraGraph& graph) const;
    };
struct OpMaxPool {
void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
static constexpr const char* getOpType() { return "MaxPool"; }
std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
const std::vector<std::string> getAttrsString() const {
    return {
        "kernel_shape",
"auto_pad",
"ceil_mode",
"dilations",
"pads",
"storage_order",
"strides"
    };
}

        std::string input;

        std::string output;
        std::string indices;

    std::vector<int64_t> kernel_shape;
    std::string auto_pad = std::string("NOTSET");
    int64_t ceil_mode = int64_t(0);
    std::vector<int64_t> dilations = std::vector<int64_t>({1, 1});
    std::vector<int64_t> pads = std::vector<int64_t>({0, 0});
    int64_t storage_order = int64_t(0);
    std::vector<int64_t> strides = std::vector<int64_t>({1, 1});

bool verify(const Core::BebraGraph& graph) const;
    };
struct OpReduceMean {
void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
static constexpr const char* getOpType() { return "ReduceMean"; }
std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
const std::vector<std::string> getAttrsString() const {
    return {
        "axes",
"keepdims"
    };
}

        std::string input;
        std::string axes_t;

        std::string output;

    std::vector<int64_t> axes;
    int64_t keepdims = int64_t(1);

bool verify(const Core::BebraGraph& graph) const;
    };
struct OpReshape {
void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
static constexpr const char* getOpType() { return "Reshape"; }
std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
const std::vector<std::string> getAttrsString() const {
    return {
        
    };
}

        std::string input;
        std::string shape;

        std::string output;


bool verify(const Core::BebraGraph& graph) const;
    };
struct OpSigmoid {
void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
static constexpr const char* getOpType() { return "Sigmoid"; }
std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
const std::vector<std::string> getAttrsString() const {
    return {
        
    };
}

        std::string input;

        std::string output;


bool verify(const Core::BebraGraph& graph) const;
    };
struct OpFlatten {
void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
static constexpr const char* getOpType() { return "Flatten"; }
std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
const std::vector<std::string> getAttrsString() const {
    return {
        "axis"
    };
}

        std::string input;

        std::string output;

    int64_t axis = int64_t(1);

bool verify(const Core::BebraGraph& graph) const;
    };
} // end of Ops :0
} // end of Bebra :0
