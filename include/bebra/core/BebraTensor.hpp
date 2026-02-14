// BebraTensor.hpp

#pragma once
#include <string>

namespace Bebra {
namespace Core {
struct BebraTensor {

    std::string name_;
    std::vector<int64_t> shape_;
    std::vector<int8_t> data_;
    int64_t dtype;

    BebraTensor(const std::string& name) : name_(name) {
        std::cout << "Created tensor with name: " << name << std::endl;

    }

};

} // end of Core :0
} // end of Bebra :0
