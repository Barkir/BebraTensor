#include "bebra/passes/ShapeInferencePass.hpp"
#include "bebra/ops/BebraOperators.hpp"

namespace Bebra {
namespace Pass {

// const std::unordered_map<std::string, ShapeInferenceFunc> inferenceMap = {
//     {"Add", ShapeInferencePass::inferAdd}
// };

bool ShapeInferencePass::run(Core::BebraGraph& graph) {
    bool success = true;
    graph_ = &graph;

    // sorry za copipast, maybe tut codegen nuzhen ili hash-table
    // for (auto&& node : graph.nodes_) {
    //     if (node.op_type_ == "Conv") {
    //         success = inferConv(node);
    //     } else if (node.op_type_ == "Mul") {
    //         success = inferMul(node);
    //     } else if (node.op_type_ == "MaxPool") {
    //         // success = inferMaxPool(node);
    //     } else if (node.op_type_ == "MatMul") {
    //         // success = inferMatMul(node);
    //     } else if (node.op_type_ == "Add") {
    //         success = inferAdd(node);
    //     } else if (node.op_type_ == "Relu") {
    //         // success = inferRelu(node);
    //     } else if (node.op_type_ == "Gemm") {
    //         // success = inferGemm(node);
    //     }
    // }

    return success;
}

bool ShapeInferencePass::inferConv(Core::BebraNode& node) {

//     assert(graph_ != nullptr);
//
//     auto&& it1 = graph_->tensor_map_.find(node.inputs_[0]);
//     if (it1 == graph_->tensor_map_.end()) {
//         return false;
//     }
//     auto&& input_tensor = it1->second;
//
//     auto&& it2 = graph_->tensor_map_.find(node.inputs_[1]);
//     if (it2 == graph_->tensor_map_.end()) {
//         return false;
//     }
//     auto&& weight_tensor = it2->second;
//
//     if (input_tensor.shape_.size() != 4) return false;
//
//     Ops::OpConv conv(&node);
//     auto strides = conv.strides();
//     auto pads = conv.pads();
//     auto dilations = conv.dilations();
//
//     int64_t N = input_tensor.shape_[0];
//     int64_t C_out = weight_tensor.shape_[0];
//     int64_t H = input_tensor.shape_[2];
//     int64_t W = input_tensor.shape_[3];
//     int64_t KH = weight_tensor.shape_[2];
//     int64_t KW = weight_tensor.shape_[3];
//
//     int64_t H_out = (H + pads[0] + pads[2] -
//                      dilations[0] * (KH - 1) - 1) / strides[0] + 1;
//     int64_t W_out = (W + pads[1] + pads[3] -
//                      dilations[1] * (KW - 1) - 1) / strides[1] + 1;
//
//     for (auto&& out_name : node.outputs_) {
//         auto&& out_it = graph_->tensor_map_.find(out_name);
//         if (out_it == graph_->tensor_map_.end()) {
//             return false;
//         }
//         out_it->second.shape_ = {N, C_out, H_out, W_out};
//     }
//
//     return true;
}

bool ShapeInferencePass::inferMul(Core::BebraNode& node) {

//     assert(graph_ != nullptr);
//
//     auto&& input_tensor = graph_->tensor_map_.find(node.inputs_[0]);
//     if (input_tensor == graph_->tensor_map_.end()) {
//         return false;
//     }
//
//     for (auto&& out: node.outputs_) {
//         auto&& out_it = graph_->tensor_map_.find(out);
//         if (out_it == graph_->tensor_map_.end()) {
//             return false;
//         }
//
//         out_it->second.shape_ = input_tensor->second.shape_;
//     }
//
//     return true;
}

bool ShapeInferencePass::inferAdd(Core::BebraNode& node) {

//     assert(graph_ != nullptr);
//
//     auto&& input_tensor = graph_->tensor_map_.find(node.inputs_[0]);
//
//     if (input_tensor == graph_->tensor_map_.end()) {
//         return false;
//     }
//
//     for (auto&& out: node.outputs_) {
//         auto&& out_it = graph_->tensor_map_.find(out);
//         if (out_it == graph_->tensor_map_.end()) {
//             return false;
//         }
//
//         out_it->second.shape_ = input_tensor->second.shape_;
//     }
//
//     return true;
}




} // end of Pass :0
} // end of Bebra :0
