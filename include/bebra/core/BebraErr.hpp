// BebraErr.hpp

#pragma once

#include "BebraColors.hpp"
#include <stdexcept>
#include <string>

namespace Bebra {
namespace Core {

struct BebraErr : public std::runtime_error {
    explicit BebraErr(const std::string& msg) : std::runtime_error(BG_BRIGHT_RED + msg + BEBRA_RESET) {}
};

} // namespace Core
} // namespace Bebra
