// BebraErr.hpp

#pragma once

#include "BebraColors.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace Bebra {
namespace Core {

struct BebraErr : public std::runtime_error {
    explicit BebraErr(const std::string& msg) : std::runtime_error(BG_BRIGHT_RED + msg + BEBRA_RESET) {}
};

struct BebraWarn {
    explicit BebraWarn(const std::string& msg) {
        std::cout << BG_BRIGHT_YELLOW << msg << BEBRA_RESET << std::endl;
    }
};

struct BebraGreen {
    explicit BebraGreen(const std::string& msg) {
        std::cout << BOLD_GREEN << msg << BEBRA_RESET << std::endl;
    }
};

} // namespace Core
} // namespace Bebra
