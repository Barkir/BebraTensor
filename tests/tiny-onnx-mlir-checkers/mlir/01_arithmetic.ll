#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, 0)>
module {
  func.func @main(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> {
    %cst = arith.constant dense<2.000000e+00> : tensor<1x1x1x1xf32>
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<1x3x32x32xf32>
    %0 = tensor.empty() : tensor<1x3x32x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_2 : tensor<1x3x32x32xf32>) outs(%0 : tensor<1x3x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.minimumf %in, %cst_0 : f32
      %4 = arith.maximumf %3, %cst_1 : f32
      linalg.yield %4 : f32
    } -> tensor<1x3x32x32xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %cst : tensor<1x3x32x32xf32>, tensor<1x1x1x1xf32>) outs(%0 : tensor<1x3x32x32xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %3 = arith.mulf %in, %in_3 : f32
      linalg.yield %3 : f32
    } -> tensor<1x3x32x32xf32>
    return %2 : tensor<1x3x32x32xf32>
  }
}
