#include "BebraType.hpp"

namespace Bebra::Core {
BebraType OnnxDtypeToBebra(int32_t onnx_dtype) {
    switch (onnx_dtype) {
        case onnx::TensorProto::FLOAT:      return BebraType::FLOAT;
        case onnx::TensorProto::DOUBLE:     return BebraType::DOUBLE;
        case onnx::TensorProto::INT32:      return BebraType::INT32;
        case onnx::TensorProto::INT64:      return BebraType::INT64;
        case onnx::TensorProto::UINT8:      return BebraType::UINT8;
        case onnx::TensorProto::INT8:       return BebraType::INT8;
        case onnx::TensorProto::BOOL:       return BebraType::BOOL;
        case onnx::TensorProto::FLOAT16:    return BebraType::FLOAT16;
        default:                      return BebraType::UNDEF;
    }
}

onnx::TensorProto::DataType BebraToOnnxDtype(BebraType type) {
    switch(type) {
        case BebraType::FLOAT: return onnx::TensorProto::FLOAT;
        case BebraType::DOUBLE: return onnx::TensorProto::DOUBLE;
        case BebraType::FLOAT16: return onnx::TensorProto::FLOAT16;
        case BebraType::INT32: return onnx::TensorProto::INT32;
        case BebraType::INT64: return onnx::TensorProto::INT64;
        case BebraType::INT8: return onnx::TensorProto::INT8;
        case BebraType::BOOL: return onnx::TensorProto::BOOL;
        default: return onnx::TensorProto::UNDEFINED;
    }
}

std::ostream& operator<<(std::ostream& s, const BebraType& type) {
    switch(type) {
        case BebraType::BOOL: s << "BOOL"; break;
        case BebraType::DOUBLE: s << "DOUBLE"; break;
        case BebraType::FLOAT16: s << "FLOAT16"; break;
        case BebraType::FLOAT: s << "FLOAT"; break;
        case BebraType::INT32: s << "INT32"; break;
        case BebraType::INT64: s << "INT64"; break;
        case BebraType::INT8: s << "INT8"; break;
        case BebraType::UNDEF: s << "UNDEF"; break;
        default: s << "UNDEF"; break;
    }

    return s;
}
}
