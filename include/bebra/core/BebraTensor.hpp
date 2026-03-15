// BebraTensor.hpp

#pragma once
#include <iostream>
#include <memory>
#include <string>

#include "bebra/core/BebraErr.hpp"
#include "bebra/core/BebraType.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include "onnx_proto/onnx.proto3.pb.h"

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
    [[nodiscard]] bool hasDynamicShape() const {
        return isDynamicShape;
    }
    [[nodiscard]] bool hasShape() const {
        return shape_.size() != 0;
    }
    [[nodiscard]] bool hasData() const noexcept {
        return !data_.empty();
    }
    [[nodiscard]] const std::vector<int8_t>& data() const noexcept {
        return data_;
    }
    [[nodiscard]] const std::string& getName() const {
        return name_;
    }
    [[nodiscard]] std::vector<int64_t> getShape() const {
        return shape_;
    }
    [[nodiscard]] BebraType getBebraType() const {
        return dtype;
    }
    [[nodiscard]] onnx::TensorProto::DataType getOnnxType() const {
        return BebraToOnnxDtype(dtype);
    }

    template <typename T>
    [[nodiscard]] const T* dataAs() const noexcept {
        return reinterpret_cast<const T*>(data_.data());
    }

    template <typename T>
    [[nodiscard]] bool hasCorrectDataType() const noexcept {
        return data_.size() % sizeof(T) == 0;
    }

    void setShape(std::vector<int64_t>&& shape) noexcept {
        shape_ = std::move(shape);
    }

    void assignDataByType(const onnx::TensorProto& tensor, BebraType t) {
        switch (t) {
            case BebraType::FLOAT: {
                auto dfloat = tensor.float_data();
                auto nbytes = tensor.float_data_size() * sizeof(float);
                data_.resize(nbytes);
                std::memcpy(data_.data(), dfloat.data(), nbytes);
                break;
            }

            case BebraType::DOUBLE: {
                auto ddouble = tensor.double_data();
                auto nbytes = tensor.double_data_size() * sizeof(double);
                data_.resize(nbytes);
                std::memcpy(data_.data(), ddouble.data(), nbytes);
                break;
            }

            case BebraType::INT64: {
                auto dint64 = tensor.int64_data();
                auto nbytes = tensor.int64_data_size() * sizeof(int64_t);
                data_.resize(nbytes);
                std::memcpy(data_.data(), dint64.data(), nbytes);
                break;
            }

            case BebraType::INT32: {
                auto dint32 = tensor.int32_data();
                auto nbytes = tensor.int32_data_size() * sizeof(int32_t);
                data_.resize(nbytes);
                std::memcpy(data_.data(), dint32.data(), nbytes);
                break;
            }

            default: {
                throw BebraErr("unsupported BebraType in assignDataByType");
            }
        }
        if (data_.empty()) {
            BebraWarn("data_ is empty after assignment !!! ");
        }
    }
};

} // namespace Core
} // namespace Bebra
