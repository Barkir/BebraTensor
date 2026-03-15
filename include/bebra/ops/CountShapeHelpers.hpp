#pragma once

#include <vector>
#include <cstdint>
using shapeVec = std::vector<int64_t>;
namespace Bebra::Ops {
shapeVec calculateMatMulShape(shapeVec shapeA, shapeVec shapeB);
shapeVec calculateBroadcastShape(shapeVec shapeA, shapeVec shapeB);
shapeVec calculateReduceShape(shapeVec inputShape, shapeVec axes, int64_t keepdims);
shapeVec calculateFlattenShape(shapeVec inputShape, int64_t axis);
}
