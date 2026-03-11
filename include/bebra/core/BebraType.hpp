#pragma once

#include <string>

#include "onnx_proto/onnx.proto3.pb.h"

namespace Bebra::Core {

enum class BebraType { FLOAT, DOUBLE, INT32, INT64, UINT8, INT8, BOOL, FLOAT16, UNDEF };

BebraType OnnxDtypeToBebra(int32_t onnx_dtype);
onnx::TensorProto::DataType BebraToOnnxDtype(BebraType type);
std::ostream& operator<<(std::ostream& s, const BebraType& type);

} // namespace Bebra::Core
