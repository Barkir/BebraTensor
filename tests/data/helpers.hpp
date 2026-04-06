#pragma once

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

static std::string get_model_path(std::string path) {
    return std::filesystem::path(TEST_DATA_DIR) / "tests/tiny-onnx-mlir-checkers" / path;
}

static std::string get_model_path2(std::string path) {
    return std::filesystem::path(TEST_DATA_DIR) / "tests" / path;
}

static std::string get_ll_path(std::string path) {
    auto toSys = std::filesystem::path(path);
    auto name = toSys.stem();
    auto ll_name = name.string() + ".ll";
    return std::filesystem::path(TEST_DATA_DIR) / "tests/tiny-onnx-mlir-checkers/mlir" / ll_name;
}

static std::string get_checker_path(std::string path) {
  auto toSys = std::filesystem::path(path);
  auto checker_res = toSys.stem();
  llvm::errs() << "res_stem = " << checker_res << "\n";
  auto new_checker = checker_res.string() + "_checker.ll";
  auto new_path = std::filesystem::path(TEST_DATA_DIR) / "tests/tiny-onnx-mlir-checkers/checkers" / new_checker;

  llvm::errs() << "got new_path " << new_path << "\n";

  return new_path;
}


static void VerifyIR(std::string res, std::string checkFilePath) {
    auto command = "echo '" + res + "' | FileCheck " + checkFilePath;
    EXPECT_EQ(std::system(command.c_str()), 0);
}
