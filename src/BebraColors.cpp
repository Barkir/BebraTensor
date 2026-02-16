// BebraColors.cpp
//NOTE - AI-generated

#include "bebra/core/BebraColors.hpp"

std::string getNodeColor(const std::string& nodeName) {
    if (nodeName == "Conv") return BEBRA_CONV_NODE;
    if (nodeName == "ConvTranspose") return BEBRA_CONV_NODE;
    if (nodeName == "Relu") return BEBRA_RELU_NODE;
    if (nodeName == "LeakyRelu") return BEBRA_LEAKYRELU_NODE;
    if (nodeName == "MaxPool") return BEBRA_MAXPOOL_NODE;
    if (nodeName == "AveragePool") return BEBRA_AVGPOOL_NODE;
    if (nodeName == "GlobalAveragePool") return BEBRA_GLOBALPOOL_NODE;
    if (nodeName == "GlobalMaxPool") return BEBRA_GLOBALPOOL_NODE;
    if (nodeName == "BatchNormalization") return BEBRA_BATCHNORM_NODE;
    if (nodeName == "Dropout") return BEBRA_DROPOUT_NODE;
    if (nodeName == "Softmax") return BEBRA_SOFTMAX_NODE;
    if (nodeName == "Gemm") return BEBRA_GEMM_NODE;
    if (nodeName == "MatMul") return BEBRA_MATMUL_NODE;
    if (nodeName == "Flatten") return BEBRA_FLATTEN_NODE;
    if (nodeName == "Reshape") return BEBRA_RESHAPE_NODE;
    if (nodeName == "Transpose") return BEBRA_TRANSPOSE_NODE;
    if (nodeName == "Concat") return BEBRA_CONCAT_NODE;
    if (nodeName == "Split") return BEBRA_SPLIT_NODE;
    if (nodeName == "Add") return BEBRA_ADD_NODE;
    if (nodeName == "Mul") return BEBRA_MUL_NODE;
    if (nodeName == "Sigmoid") return BEBRA_SIGMOID_NODE;
    if (nodeName == "Tanh") return BEBRA_TANH_NODE;
    if (nodeName == "Input") return BEBRA_INPUT_NODE;
    if (nodeName == "Output") return BEBRA_OUTPUT_NODE;
    return BEBRA_DEFAULT_NODE;
}
