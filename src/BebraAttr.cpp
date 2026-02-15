// BebraAttr.cpp

#include "bebra/core/BebraAttr.hpp"
#include "bebra/core/BebraErr.hpp"

namespace Bebra {
namespace Core {

Attr parseAttr(const onnx::AttributeProto& attr) {
    switch (attr.type()) {
        case onnx::AttributeProto::INT:
            return attr.i();

        case onnx::AttributeProto::INTS:
            return std::vector<int64_t>(attr.ints().begin(), attr.ints().end());

        case onnx::AttributeProto::FLOAT:
            return attr.f();

        case onnx::AttributeProto::FLOATS:
            return std::vector<float>(attr.floats().begin(), attr.floats().end());

        case onnx::AttributeProto::STRING:
            return attr.s();

        case onnx::AttributeProto::STRINGS:
            return std::vector<std::string>(attr.strings().begin(), attr.strings().end());

        case onnx::AttributeProto::TENSOR:
            return BebraTensor(attr.name());


        default:
            throw BebraErr("Unknown AttributeType: " + std::to_string(attr.type()));

        //TODO BebraTensor and stuff...

    }
}

} // end of Core :0
} // end of Bebra :0
