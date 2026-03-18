#include "bebra/core/BebraTensor.hpp"

namespace Bebra::Core {

BebraTensor::BebraTensor(const onnx::TensorProto& tensor)
    : name_(tensor.name()), dtype(OnnxDtypeToBebra(tensor.data_type())) {
    auto&& dsize = tensor.dims_size();
    shape_.reserve(static_cast<size_t>(dsize));
    for (int i = 0; i < dsize; ++i) {
        shape_.push_back(tensor.dims(i));
    }

    assignDataByType(tensor, dtype);

    std::cout << "-----------------------------------------\n";
    std::cout << "Created tensor: \"" << name_ << "\"\n";
    std::cout << "  Shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i] << (i + 1 < shape_.size() ? ", " : "");
    }
    std::cout << "]\n";
    std::cout << "  Dtype: " << static_cast<int>(dtype) << "\n";
    std::cout << "  Data size: " << data_.size() << " bytes\n";
    std::cout << "-----------------------------------------\n";
}

BebraTensor::BebraTensor(const std::string& name,
                         const std::vector<int64_t> shape,
                         const std::vector<int8_t> data,
                         BebraType dtype_)
    : name_(name), shape_(shape), data_(data), dtype(dtype_) {
    std::cout << "-----------------------------------------" << "\n";
    std::cout << "Created tensor with name" << name << "\n";
    std::cout << "ndims = " << shape_.size() << "\n";
    std::cout << "dtype = " << static_cast<int>(dtype) << "\n";
    std::cout << "-----------------------------------------" << "\n";
}

BebraTensor::BebraTensor(const onnx::ValueInfoProto& value_info) : name_(value_info.name()) {
    auto&& type = value_info.type();

    if (type.has_tensor_type()) {
        auto&& tensor_type = type.tensor_type();

        dtype = OnnxDtypeToBebra(tensor_type.elem_type());

        if (tensor_type.has_shape()) {
            const auto& shape = tensor_type.shape();
            for (const auto& dim : shape.dim()) {
                if (dim.has_dim_value()) {
                    shape_.push_back(dim.dim_value());
                } else if (dim.has_dim_param()) {
                    // shape can be dynamic, so we mark it by -1
                    // example: batch_size

                    shape_.push_back(-1);
                    isDynamicShape = true;
                    std::cout << "  [Dynamic dim: " << dim.dim_param() << "]\n";
                } else {
                    shape_.push_back(-1);
                    isDynamicShape = true;
                }
            }
        }
    } else {
        std::cerr << "Warning: ValueInfoProto '" << name_ << "' is not a tensor type\n";
    }
}
} // namespace Bebra::Core
