// BebraAttr.cpp

#include "bebra/core/BebraAttr.hpp"
#include "bebra/core/BebraErr.hpp"

namespace Bebra {
namespace Core {

AttrVal parseAttr(const onnx::AttributeProto& attr) {
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

        case onnx::AttributeProto::TENSOR: {
            std::vector<BebraTensor> tensors;
            tensors.reserve(static_cast<long unsigned int>(attr.tensors_size()));
            for (auto&& t : attr.tensors()) {
                tensors.push_back(BebraTensor(t));
            }
            return tensors;
        }

        default:
            throw BebraErr("Unknown AttributeType: " + std::to_string(static_cast<unsigned int>(attr.type())));

            // TODO BebraTensor and stuff...
    }
}

} // namespace Core
} // namespace Bebra
