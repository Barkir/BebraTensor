#include "mlir/PassPipelineMLIR.hpp"

std::unique_ptr<mlir::Pass> createConvertMatmulToBlasLibraryCallPass() {
  return std::make_unique<mlir::tutorial::ConvertMatmulToBlasLibraryCallPass>();
}

void linalgToBufferizationPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // One-shot bufferize
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);
}

void BufferizationToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // CRITICAL: Replace matmuls with BLAS calls AFTER bufferization but BEFORE
  // other LLVM conversions
  manager.addPass(createConvertMatmulToBlasLibraryCallPass());

  // Convert remaining linalg ops to loops
  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // Standard LLVM lowering pipeline
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::affine::createLoopFusionPass());
  manager.addPass(mlir::affine::createAffineVectorize());
  manager.addPass(mlir::createSCFToControlFlowPass());

  // Convert to LLVM - order matters here
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertMathToLLVMPass());
  manager.addPass(
      mlir::createConvertMathToLibmPass()); // For bert model to lower math.erf
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}

// int main(int argc, char **argv) {
//   mlir::DialectRegistry registry;
//   mlir::registerAllDialects(registry);
//   mlir::registerAllPasses();
//
//   mlir::PassRegistration<mlir::tutorial::ConvertMatmulToBlasLibraryCallPass>();
//
//   mlir::PassPipelineRegistration<>(
//       "linalg-to-bufferization",
//       "Run passes to lower the linalg dialect to bufferization",
//       linalgToBufferizationPipelineBuilder);
//
//   mlir::PassPipelineRegistration<>(
//       "bufferization-to-llvm", "Run passes to lower bufferized code to LLVM",
//       BufferizationToLLVMPipelineBuilder);
//
//   return mlir::asMainReturnCode(
//       mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
// }
