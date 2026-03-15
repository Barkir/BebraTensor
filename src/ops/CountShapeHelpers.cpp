#include "bebra/ops/CountShapeHelpers.hpp"
#include "bebra/core/BebraErr.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace Bebra::Ops {
shapeVec calculateBroadcastShape(shapeVec shapeA, shapeVec shapeB) {
    size_t rankA = shapeA.size();
    size_t rankB = shapeB.size();
    size_t outRank = std::max(rankA, rankB);
    shapeVec result(outRank);

    for (size_t i = 0; i < outRank; ++i) {
        int64_t dimA = (i < rankA) ? shapeA[rankA - 1 - i] : 1;
        int64_t dimB = (i < rankB) ? shapeB[rankB - 1 - i] : 1;

        if (dimA != dimB && dimA != 1 && dimB != 1) {
            throw Core::BebraErr("Incompatible shapes for broadcasting");
        }
        result[outRank - 1 - i] = std::max(dimA, dimB);
    }
    return result;
}

shapeVec calculateMatMulShape(shapeVec shapeA, shapeVec shapeB) {
    if (shapeA.size() < 2 || shapeB.size() < 2) {
        throw Core::BebraErr("MatMul requires at least 2D tensors");
    }

    size_t rankA = shapeA.size();
    size_t rankB = shapeB.size();

    if (shapeA.back() != shapeB[rankB - 2]) {
        throw Core::BebraErr("MatMul: Inner dimensions must match");
    }

    shapeVec batchA(shapeA.begin(), shapeA.end() - 2);
    shapeVec batchB(shapeB.begin(), shapeB.end() - 2);
    shapeVec result = calculateBroadcastShape(batchA, batchB);

    result.push_back(shapeA[rankA - 2]); // M
    result.push_back(shapeB.back());     // N

    return result;
}

shapeVec calculateReduceShape(shapeVec inputShape, shapeVec axes, int64_t keepdims) {
    shapeVec result;
    std::vector<int64_t> normAxes = axes;
    for (auto& axis : normAxes) {
        if (axis < 0)
            axis += inputShape.size();
    }

    for (size_t i = 0; i < inputShape.size(); ++i) {
        bool isReduced = std::find(normAxes.begin(), normAxes.end(), i) != normAxes.end();

        if (isReduced) {
            if (keepdims)
                result.push_back(1);
        } else {
            result.push_back(inputShape[i]);
        }
    }

    if (result.empty())
        return {1};
    return result;
}

shapeVec calculateFlattenShape(shapeVec inputShape, int64_t axis) {
    if (axis < 0)
        axis += inputShape.size();

    int64_t dim0 = 1;
    for (int i = 0; i < axis; ++i) {
        dim0 *= inputShape[i];
    }

    int64_t dim1 = 1;
    for (size_t i = axis; i < inputShape.size(); ++i) {
        dim1 *= inputShape[i];
    }

    return {dim0, dim1};
}
} // namespace Bebra::Ops
