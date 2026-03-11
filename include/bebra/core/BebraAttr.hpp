// BebraAttr.hpp

#pragma once

/* FOR REFERENCE ONLY

enum AttributeProto_AttributeType : int {
  AttributeProto_AttributeType_UNDEFINED = 0,
  AttributeProto_AttributeType_FLOAT = 1,
  AttributeProto_AttributeType_INT = 2,
  AttributeProto_AttributeType_STRING = 3,
  AttributeProto_AttributeType_TENSOR = 4,
  AttributeProto_AttributeType_GRAPH = 5,
  AttributeProto_AttributeType_SPARSE_TENSOR = 11,
  AttributeProto_AttributeType_TYPE_PROTO = 13,
  AttributeProto_AttributeType_FLOATS = 6,
  AttributeProto_AttributeType_INTS = 7,
  AttributeProto_AttributeType_STRINGS = 8,
  AttributeProto_AttributeType_TENSORS = 9,
  AttributeProto_AttributeType_GRAPHS = 10,
  AttributeProto_AttributeType_SPARSE_TENSORS = 12,
  AttributeProto_AttributeType_TYPE_PROTOS = 14,
  AttributeProto_AttributeType_AttributeProto_AttributeType_INT_MIN_SENTINEL_DO_NOT_USE_ =
std::numeric_limits<int32_t>::min(),
  AttributeProto_AttributeType_AttributeProto_AttributeType_INT_MAX_SENTINEL_DO_NOT_USE_ =
std::numeric_limits<int32_t>::max()
};

*/

#include <string>
#include <variant>
#include <vector>

#include "bebra/core/BebraErr.hpp"
#include "bebra/core/BebraTensor.hpp"
#include "onnx_proto/onnx.proto3.pb.h"

namespace Bebra {
namespace Core {

using AttrVal = std::variant<float,
                             int64_t,
                             std::string,

                             std::vector<float>,
                             std::vector<int64_t>,
                             std::vector<std::string>,

                             BebraTensor,
                             std::vector<BebraTensor>

                             // TODO also add BebraTensor, BebraGraph :'
                             >;

class Attr {
    AttrVal val_;

public:
    template <typename T>
    Attr(T val) : val_(val) {}

    template <typename T>
    const T& getValRef() const {
        return std::get<T>(val_);
    }

    template <typename T>
    T& getValRef() {
        return std::get<T>(val_);
    }

    template <typename T>
    const T getVal() const {
        return std::get<T>(val_);
    }

    template <typename T>
    T getVal() {
        return std::get<T>(val_);
    }
};

AttrVal parseAttr(const onnx::AttributeProto& attr);

} // namespace Core
} // namespace Bebra
