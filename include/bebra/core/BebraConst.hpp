#pragma once

static const std::string nigga_model = "../third_party/mnist-8.onnx";
static const std::string other_model = "../third_party/resnet50-v1-7.onnx";
static const std::string available_flags = "\t`--dump`\n\t`--to-mlir`\n\t--help";

static const std::string HELPING_MESSAGE = "tensor compiler by Barkir \n"
    "Usage:\n"
    "\tbebra_tensor [flags]\n"
    "\tFlags:\n"
    "\t--to-mlir \t convert onnx to mlir\n"
    "\t--dump    \t dump onnx to .dot file\n";
