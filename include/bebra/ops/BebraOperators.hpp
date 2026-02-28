#pragma once
#include <string>
#include "bebra/core/BebraNode.hpp"
#include "bebra/core/BebraErr.hpp"
#include "bebra/core/BebraColors.hpp"
namespace Bebra {
namespace Ops {
struct OpConv {
    const Core::BebraNode* node_;

    explicit OpConv(const Core::BebraNode* node) : node_(node) {
        if (node_->op_type_ != "Conv") {
            throw Core::BebraErr("Not a Conv node...");
        }
        std::cout << UNDERLINE_GREEN "Got Conv node!" RESET << std::endl;
    }
    std::vector<int64_t> kernel_shape() const  {
    auto it = node_->attrs_.find("kernel_shape");
    if (it == node_->attrs_.end()) {
        throw Core::BebraErr("Missing kernel_shape at Conv!");
    }

    std::cout << FG_GOLD_RGB "Got kernel_shape attr" << RESET << std::endl;
    return it->second.getValRef<std::vector<int64_t>>();
}

int64_t group() const  {
    auto it = node_->attrs_.find("group");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got group attr" RESET<< " by default." << std::endl;
        return int64_t(1);
    }

    std::cout << FG_GOLD_RGB "Got group attr." RESET << std::endl;
    return it->second.getValRef<int64_t>();
}

std::vector<int64_t> dilations() const  {
    auto it = node_->attrs_.find("dilations");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got dilations attr" RESET<< " by default." << std::endl;
        return std::vector<int64_t>({1, 1});
    }

    std::cout << FG_GOLD_RGB "Got dilations attr." RESET << std::endl;
    return it->second.getValRef<std::vector<int64_t>>();
}

std::vector<int64_t> pads() const  {
    auto it = node_->attrs_.find("pads");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got pads attr" RESET<< " by default." << std::endl;
        return std::vector<int64_t>({0, 0});
    }

    std::cout << FG_GOLD_RGB "Got pads attr." RESET << std::endl;
    return it->second.getValRef<std::vector<int64_t>>();
}

std::vector<int64_t> strides() const  {
    auto it = node_->attrs_.find("strides");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got strides attr" RESET<< " by default." << std::endl;
        return std::vector<int64_t>({1, 1});
    }

    std::cout << FG_GOLD_RGB "Got strides attr." RESET << std::endl;
    return it->second.getValRef<std::vector<int64_t>>();
}

    };
struct OpGemm {
    const Core::BebraNode* node_;

    explicit OpGemm(const Core::BebraNode* node) : node_(node) {
        if (node_->op_type_ != "Gemm") {
            throw Core::BebraErr("Not a Gemm node...");
        }
        std::cout << UNDERLINE_GREEN "Got Gemm node!" RESET << std::endl;
    }
    float alpha() const  {
    auto it = node_->attrs_.find("alpha");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got alpha attr" RESET<< " by default." << std::endl;
        return float(1.0f);
    }

    std::cout << FG_GOLD_RGB "Got alpha attr." RESET << std::endl;
    return it->second.getValRef<float>();
}

float beta() const  {
    auto it = node_->attrs_.find("beta");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got beta attr" RESET<< " by default." << std::endl;
        return float(1.0f);
    }

    std::cout << FG_GOLD_RGB "Got beta attr." RESET << std::endl;
    return it->second.getValRef<float>();
}

int64_t transA() const  {
    auto it = node_->attrs_.find("transA");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got transA attr" RESET<< " by default." << std::endl;
        return int64_t(0);
    }

    std::cout << FG_GOLD_RGB "Got transA attr." RESET << std::endl;
    return it->second.getValRef<int64_t>();
}

int64_t transB() const  {
    auto it = node_->attrs_.find("transB");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got transB attr" RESET<< " by default." << std::endl;
        return int64_t(0);
    }

    std::cout << FG_GOLD_RGB "Got transB attr." RESET << std::endl;
    return it->second.getValRef<int64_t>();
}

    };
struct OpAdd {
    const Core::BebraNode* node_;

    explicit OpAdd(const Core::BebraNode* node) : node_(node) {
        if (node_->op_type_ != "Add") {
            throw Core::BebraErr("Not a Add node...");
        }
        std::cout << UNDERLINE_GREEN "Got Add node!" RESET << std::endl;
    }
    
    };
struct OpRelu {
    const Core::BebraNode* node_;

    explicit OpRelu(const Core::BebraNode* node) : node_(node) {
        if (node_->op_type_ != "Relu") {
            throw Core::BebraErr("Not a Relu node...");
        }
        std::cout << UNDERLINE_GREEN "Got Relu node!" RESET << std::endl;
    }
    
    };
struct OpMul {
    const Core::BebraNode* node_;

    explicit OpMul(const Core::BebraNode* node) : node_(node) {
        if (node_->op_type_ != "Mul") {
            throw Core::BebraErr("Not a Mul node...");
        }
        std::cout << UNDERLINE_GREEN "Got Mul node!" RESET << std::endl;
    }
    
    };
struct OpMatMul {
    const Core::BebraNode* node_;

    explicit OpMatMul(const Core::BebraNode* node) : node_(node) {
        if (node_->op_type_ != "MatMul") {
            throw Core::BebraErr("Not a MatMul node...");
        }
        std::cout << UNDERLINE_GREEN "Got MatMul node!" RESET << std::endl;
    }
    
    };
struct OpMaxPool {
    const Core::BebraNode* node_;

    explicit OpMaxPool(const Core::BebraNode* node) : node_(node) {
        if (node_->op_type_ != "MaxPool") {
            throw Core::BebraErr("Not a MaxPool node...");
        }
        std::cout << UNDERLINE_GREEN "Got MaxPool node!" RESET << std::endl;
    }
    std::vector<int64_t> kernel_shape() const  {
    auto it = node_->attrs_.find("kernel_shape");
    if (it == node_->attrs_.end()) {
        throw Core::BebraErr("Missing kernel_shape at MaxPool!");
    }

    std::cout << FG_GOLD_RGB "Got kernel_shape attr" << RESET << std::endl;
    return it->second.getValRef<std::vector<int64_t>>();
}

std::string auto_pad() const  {
    auto it = node_->attrs_.find("auto_pad");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got auto_pad attr" RESET<< " by default." << std::endl;
        return std::string("NOTSET");
    }

    std::cout << FG_GOLD_RGB "Got auto_pad attr." RESET << std::endl;
    return it->second.getValRef<std::string>();
}

int64_t ceil_mode() const  {
    auto it = node_->attrs_.find("ceil_mode");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got ceil_mode attr" RESET<< " by default." << std::endl;
        return int64_t(0);
    }

    std::cout << FG_GOLD_RGB "Got ceil_mode attr." RESET << std::endl;
    return it->second.getValRef<int64_t>();
}

std::vector<int64_t> dilations() const  {
    auto it = node_->attrs_.find("dilations");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got dilations attr" RESET<< " by default." << std::endl;
        return std::vector<int64_t>({1, 1});
    }

    std::cout << FG_GOLD_RGB "Got dilations attr." RESET << std::endl;
    return it->second.getValRef<std::vector<int64_t>>();
}

std::vector<int64_t> pads() const  {
    auto it = node_->attrs_.find("pads");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got pads attr" RESET<< " by default." << std::endl;
        return std::vector<int64_t>({0, 0});
    }

    std::cout << FG_GOLD_RGB "Got pads attr." RESET << std::endl;
    return it->second.getValRef<std::vector<int64_t>>();
}

int64_t storage_order() const  {
    auto it = node_->attrs_.find("storage_order");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got storage_order attr" RESET<< " by default." << std::endl;
        return int64_t(0);
    }

    std::cout << FG_GOLD_RGB "Got storage_order attr." RESET << std::endl;
    return it->second.getValRef<int64_t>();
}

std::vector<int64_t> strides() const  {
    auto it = node_->attrs_.find("strides");
    if (it == node_->attrs_.end()) {

    std::cout << FG_GOLD_RGB "Got strides attr" RESET<< " by default." << std::endl;
        return std::vector<int64_t>({1, 1});
    }

    std::cout << FG_GOLD_RGB "Got strides attr." RESET << std::endl;
    return it->second.getValRef<std::vector<int64_t>>();
}

    };
} // end of Ops :0
} // end of Bebra :0
