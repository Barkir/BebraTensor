// BebraTensor.hpp

#pragma once
#include <string>
#include <iostream>

#include "bebra/core/BebraType.hpp"
#include "onnx_proto/onnx.proto3.pb.h"
#include "mlir/IR/BuiltinTypes.h"

namespace Bebra {
namespace Core {
struct BebraTensor {

    std::string name_;
    std::vector<int64_t> shape_;
    std::vector<int8_t> data_;
    BebraType dtype;
    bool isDynamicShape = false;

public: // constructors && destructor
    explicit BebraTensor(const std::string& name) : name_(name) {
        std::cout << "-----------------------------------------" << std::endl;
        std::cout << "Created tensor with name: " << name << std::endl;
        std::cout << "WARNING! NO SHAPE INFO PROVIDED!" << std::endl;
        std::cout << "-----------------------------------------" << std::endl;
        isDynamicShape = true;
    }

    BebraTensor(const std::string& name,
                const std::vector<int64_t> shape,
                const std::vector<int8_t> data,
                BebraType dtype);

    explicit BebraTensor(const onnx::ValueInfoProto& value_info);
    explicit BebraTensor(const onnx::TensorProto& tensor);



    ~BebraTensor() = default;


public: // helper methods
    [[nodiscard]] bool hasDynamicShape() const { return isDynamicShape; }
    [[nodiscard]] bool hasShape() const { return shape_.size() != 0; }
    [[nodiscard]] bool hasData() const noexcept { return !data_.empty(); }
    [[nodiscard]] const std::vector<int8_t>& data() const noexcept { return data_; }
    [[nodiscard]] const std::string& getName() const { return name_; }
    [[nodiscard]] std::vector<int64_t> getShape() const { return shape_; }
    [[nodiscard]] BebraType getBebraType() const { return dtype; }
    [[nodiscard]] onnx::TensorProto::DataType getOnnxType() const { return BebraToOnnxDtype(dtype); }

    template <typename T>
    [[nodiscard]] const T* dataAs() const noexcept {
        return reinterpret_cast<const T*>(data_.data());
    }

    template<typename T>
    [[nodiscard]] bool hasCorrectDataType() const noexcept {
        return data_.size() % sizeof(T) == 0;
    }

};

} // end of Core :0
} // end of Bebra :0
