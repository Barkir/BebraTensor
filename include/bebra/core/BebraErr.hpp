// BebraErr.hpp

#pragma once

#include <string>
#include <stdexcept>
#include "BebraColors.hpp"


namespace Bebra {
namespace Core {

struct BebraErr : public std::runtime_error {
    explicit BebraErr(const std::string& msg) : std::runtime_error(BG_BRIGHT_RED + msg + BEBRA_RESET) {}
};

} // end of Core :0
} // end of Bebra :0
