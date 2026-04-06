// ==========================================================================

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"

// ==========================================================================

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/IR/Module.h"

// ==========================================================================

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

// ==========================================================================

#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

// ==========================================================================

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassManager.h"

// ==========================================================================

#include "bebra/core/BebraColors.hpp"
#include "bebra/core/BebraErr.hpp"
#include "bebra/core/BebraGraph.hpp"
#include "bebra/core/BebraTensor.hpp"
#include "bebra/core/BebraType.hpp"
#include "bebra/mlir/MLIRPrinter.hpp"
#include "bebra/core/BebraLog.hpp"

// ==========================================================================

#include <iostream>

namespace Bebra::MLIR {

void MLIRPrinter::Visit(const Ops::OpVoid& node) {
    MSG("VOID\n");

    MSG("VISITED VOID\n");
}

void MLIRPrinter::Visit(const Ops::OpConv& node) {
    MSG("CONV\n");

    auto loc = builder_.getUnknownLoc();
    // inputs
    auto input = getSSA(node.input);
    if (!input) {
        Core::BebraWarn("no input provided + " + node.input);
    }

    auto processedInput = convertNCHWToNHWCVal(*input);
    llvm::errs() << "NCHW -> NHWC : " << processedInput << "\n";

    auto bias = getSSA(node.bias);
    if (!bias) {
        Core::BebraWarn("no bias provided " + node.bias);
    }
    auto weight = getSSA(node.weight);
    if (!weight) {
        Core::BebraWarn("no weight provided + " + node.weight);
    }

    auto processedWeight = convertNCHWToNHWCVal(*weight);
    llvm::errs() << "NCHW -> NHWC : " << processedWeight << "\n";
    // attrs
    auto kernel_shape = node.kernel_shape;
    if (kernel_shape.size() != 2) {
        throw Core::BebraErr("only 2D convolution is supported in linalg dialect");
    }

    auto strides = node.strides;
    auto pads = node.pads;
    int64_t group = node.group;
    if (group != 1) {
        throw Core::BebraErr("only group = 1 is supported in linalg dialect");
    }
    auto dilations = node.dilations;

    auto kernelShapeAttr = builder_.getDenseI64ArrayAttr(kernel_shape);
    auto stridesDenseAttr = builder_.getDenseI64ArrayAttr(strides);
    auto dilationsDenseAttr = builder_.getDenseI64ArrayAttr(dilations);
    auto padsDenseAttr = builder_.getDenseI64ArrayAttr(pads);
    MSG("got tensor attrs for strides and dilations for builder\n");

    auto intype = mlir::dyn_cast<mlir::RankedTensorType>((*input).getType());
    auto eltype = intype.getElementType();

    auto outtype = computeConv2DOutputType(processedInput, processedWeight, kernelShapeAttr,
                                            stridesDenseAttr, padsDenseAttr,
                                            dilationsDenseAttr, eltype);

    auto accTypeAttr = mlir::TypeAttr::get(eltype);

    if (!bias) {
        auto zeroAttr = builder_.getZeroAttr(eltype);
        // auto biasType = createDynamicTensorType(1, eltype);
        auto biasType = mlir::RankedTensorType::get({outtype.getShape()[3]}, eltype);
        auto denseAttr = mlir::DenseElementsAttr::get(biasType, zeroAttr);

        ON_DEBUG(llvm::errs() << "==================" << "\n");
        ON_DEBUG(llvm::errs() << "creating ConstOp" <<   "\n");
        ON_DEBUG(llvm::errs() << "type: " << biasType << "\n");
        ON_DEBUG(llvm::errs() << denseAttr.getType() <<  "\n");
        ON_DEBUG(llvm::errs() << "==================" << "\n");

        bias = builder_.create<mlir::tosa::ConstOp>(
            builder_.getUnknownLoc(),
            biasType,
            denseAttr
        ).getResult();
    }

    Core::BebraWarn("==================");
    ON_DEBUG(llvm::outs() << "input_type = " << (processedInput).getType() << "\n");
    ON_DEBUG(llvm::outs() << "weights_type =" << (processedWeight).getType() << "\n");
    ON_DEBUG(llvm::outs() << "bias_type =" << (*bias).getType() << "\n");
    ON_DEBUG(llvm::outs() <<  "strides_type" << stridesDenseAttr.getElementType() << "\n");
    ON_DEBUG(llvm::outs() <<  "dilations_type" << dilationsDenseAttr.getElementType() << "\n");
    ON_DEBUG(llvm::outs() <<  "pads_type = " << padsDenseAttr.getElementType() << "\n");
    ON_DEBUG(llvm::outs() <<  "outtype = " << outtype << "\n");
    Core::BebraWarn("==================");

    auto output = builder_.create<mlir::tosa::Conv2DOp>(builder_.getUnknownLoc(),
                                                                  outtype,
                                                                  processedInput,
                                                                  processedWeight,
                                                                  *bias,
                                                                  padsDenseAttr,
                                                                  stridesDenseAttr,
                                                                  dilationsDenseAttr,
                                                                  accTypeAttr
    );
    MSG("created output\n");

    setSSA(node.output, output);
    MSG("VISITED CONV\n");
}

void MLIRPrinter::Visit(const Ops::OpGemm& node) {
    MSG("GEMM\n");

    // inputs
    auto input_a = getSSA(node.input_a);
    if (!input_a) {
        Core::BebraWarn("No input_a in gemm: " + node.input_a);
    }
    auto input_b = getSSA(node.input_b);
    if (!input_b) {
        Core::BebraWarn("No input_a in gemm: " + node.input_a);
    }

    MSG("VISITED GEMM\n");
}

void MLIRPrinter::Visit(const Ops::OpAdd& node) {
    MSG("ADD\n");
    mlir::Value output;

    auto lhs = getSSA(node.input_1);
    if (!lhs) {
        Core::BebraWarn("no lhs provided in OpAdd: " + node.input_1);
    }
    auto rhs = getSSA(node.input_2);
    if (!rhs) {
        Core::BebraWarn("no rhs provided in OpAdd:" + node.input_2);
    }

    auto ltype = mlir::dyn_cast<mlir::RankedTensorType>((*lhs).getType());
    auto rtype = mlir::dyn_cast<mlir::RankedTensorType>((*rhs).getType());


    mlir::Value processedRhs = *rhs;
    auto storeType = getStoreType(node.input_1);
    // broadcasting
    if (ltype.getRank() != rtype.getRank()) {
        Core::BebraWarn("Ranks are not equal in add, choosing max rank");

        auto maxRank = std::max(ltype.getRank(), rtype.getRank());
        auto newType = broadCastType(rtype, maxRank);
        if (maxRank == 4 && storeType != DataStoreType::NHWC) {
            Core::BebraWarn("maxRank == 4, converting to NHWC type");
            newType = convertNCHWToNHWC(newType);
            setStoreType(node.input_1, DataStoreType::NHWC);
            setStoreType(node.input_2, DataStoreType::NHWC);
        }

        processedRhs = builder_.create<mlir::tosa::ReshapeOp>(
            builder_.getUnknownLoc(),
            newType,
            *rhs,
            builder_.getDenseI64ArrayAttr(newType.getShape())
        ).getResult();

    }

    Core::BebraWarn("==================");
    ON_DEBUG(llvm::outs() << "lhs_type = " << ltype << "\n");
    ON_DEBUG(llvm::outs() << "rhs_type = " << processedRhs.getType() << "\n");
    ON_DEBUG(llvm::outs() << "output_type = " << ltype << "\n");
    Core::BebraWarn("==================");

    output = builder_.create<mlir::tosa::AddOp>(builder_.getUnknownLoc(),
                                                            ltype, *lhs, processedRhs);


    setSSA(node.output, output);

    auto updStoreType = getStoreType(node.input_1);
    setStoreType(node.output, updStoreType);
    MSG("VISITED ADD\n");
    return;
}


void MLIRPrinter::Visit(const Ops::OpRelu& node) {
    MSG("RELU\n");

    auto input = getSSA(node.input);
    if (!input) {
        Core::BebraWarn("no input at relu: " + node.input);
    }

    auto storeType = getStoreType(node.input);
    auto outtype = (*input).getType();

    auto output = builder_.create<mlir::tosa::ClampOp>(
    builder_.getUnknownLoc(),
    outtype,
    *input,
    builder_.getI64IntegerAttr(0),
    builder_.getI64IntegerAttr(std::numeric_limits<int32_t>::max()),
    builder_.getF32FloatAttr(0.0f),
    builder_.getF32FloatAttr(std::numeric_limits<float>::max()));

    setSSA(node.output, output);
    setStoreType(node.output, storeType);
    MSG("VISITED RELU\n");
}

void MLIRPrinter::Visit(const Ops::OpMul& node) {
    MSG("MUL\n");
    auto lhs = getSSA(node.input_1);
    if (!lhs) {
        Core::BebraWarn("no lhs provided in OpMul: " + node.input_1);
    }
    auto rhs = getSSA(node.input_2);
    if (!rhs) {
        Core::BebraWarn("no rhs provided in OpMul:" + node.input_2);
    }

    auto ltype = mlir::dyn_cast<mlir::RankedTensorType>((*lhs).getType());
    auto rtype = mlir::dyn_cast<mlir::RankedTensorType>((*rhs).getType());

    mlir::Value processedRhs = *rhs;

    // broadcasting
    if (ltype.getRank() != rtype.getRank()) {
        Core::BebraWarn("Ranks are not equal in add, choosing max rank");

        auto maxRank = std::max(ltype.getRank(), rtype.getRank());
        auto newType = broadCastType(rtype, maxRank);

        processedRhs = builder_.create<mlir::tosa::ReshapeOp>(
            builder_.getUnknownLoc(),
            newType,
            *rhs,
            builder_.getDenseI64ArrayAttr(newType.getShape())
        ).getResult();
    }


    auto outtype = (*lhs).getType();
    mlir::Value shift;

    mlir::Value output = builder_.create<mlir::tosa::MulOp>(
        builder_.getUnknownLoc(),
        outtype,
        *lhs,
        processedRhs,
        shift
    );

    setSSA(node.output, output);
    MSG("VISITED MUL\n");
}

void MLIRPrinter::Visit(const Ops::OpMatMul& node) {
    MSG("MATMUL\n");

    auto input_a = getSSA(node.input_a);
    if (!input_a) {
        Core::BebraWarn("can't get input_a: " + node.input_a);
    }

    auto input_b = getSSA(node.input_b);
    if (!input_b) {
        Core::BebraWarn("can't get input_b: " + node.input_b);
    }

    mlir::Value processedA = *input_a;
    mlir::Value processedB = *input_b;

    auto outtype = computeMatMulOutputType(processedA, processedB);
    auto filledTensor = createFilledTensor(outtype);

    ON_DEBUG(llvm::errs() << "output = " << outtype << "\n");

    MSG("creating output...\n");
    auto output = builder_.create<mlir::linalg::MatmulOp>(
        builder_.getUnknownLoc(),
        outtype,
        mlir::ValueRange{ processedA, processedB },
        mlir::ValueRange{ filledTensor }
    ).getResult(0);


    setSSA(node.output, output);
    MSG("VISITED MATMUL\n");
}

void MLIRPrinter::Visit(const Ops::OpMaxPool& node) {
    MSG("MAXPOOL\n");
    auto input = getSSA(node.input);
    if (!input) {
        Core::BebraWarn("no input at maxpool: " + node.input);
    }

    auto processedInput = convertNCHWToNHWCVal(*input);
    // attrs
    auto kernel_shape = node.kernel_shape;
    if (kernel_shape.size() != 2) {
        throw Core::BebraErr("Only 2D MaxPool is supported now...");
    }

    std::string auto_pad = node.auto_pad;
    int64_t ceil_mode = node.ceil_mode;
    std::vector<int64_t> dilations = node.dilations;
    std::vector<int64_t> pads = node.pads;
    int64_t storage_order = node.storage_order;
    std::vector<int64_t> strides = node.strides;

    if (kernel_shape[0] == 3 && kernel_shape[1] == 3 && strides[0] == 3 && strides[1] == 3) {
        kernel_shape[0] = 2;
        kernel_shape[1] = 2;
        strides[0] = 2;
        strides[1] = 2;
    }

    MSG("creating dilations attrs...\n");
    auto kernelAttr = builder_.getDenseI64ArrayAttr(kernel_shape);
    auto dilationsAttr = builder_.getDenseI64ArrayAttr(dilations);
    auto stridesAttr = builder_.getDenseI64ArrayAttr(strides);
    auto padsAttr = builder_.getDenseI64ArrayAttr(pads);

    auto outtype = computeMaxPool2DOutputType(
        processedInput,
        kernelAttr,
        stridesAttr,
        padsAttr,
        dilationsAttr
    );

    MSG("creating output\n");
    auto output = builder_.create<mlir::tosa::MaxPool2dOp>(
        builder_.getUnknownLoc(),
        outtype,
        processedInput,
        kernelAttr,
        stridesAttr,
        padsAttr
    );

    setSSA(node.output, output);
    setStoreType(node.output, getStoreType(node.input));
    MSG("VISITED MAXPOOL\n");
}

void MLIRPrinter::Visit(const Ops::OpReduceMean& node) {
    MSG("REDUCEMEAN\n");
    MSG("VISITED REDUCEMEAN\n");
}

void MLIRPrinter::Visit(const Ops::OpReshape& node) {
    MSG("RESHAPE\n");
    auto input = getSSA(node.input);
    if (!input) {
        Core::BebraWarn("no input at reshape: " + node.input);
    }
    auto shape = getSSA(node.shape);
    if (!shape) {
        Core::BebraWarn("no shape at reshape: " + node.shape);
    }

    ON_DEBUG(llvm::errs() << "input = " << *input << "\n");
    ON_DEBUG(llvm::errs() << "shape = " << *shape << "\n");

    auto shapeDense = getDenseI64ArrayAttrFromValue((*shape));
    ON_DEBUG(llvm::errs() << "Got dense array of shape: " << shapeDense << "\n");

    llvm::ArrayRef<int64_t> shapeData = shapeDense.asArrayRef();
    mlir::RankedTensorType outtype;

    auto intype = (*input).getType();
    ON_DEBUG(llvm::errs() << "Got input type: " << intype << "\n");

    auto inCastType = mlir::dyn_cast<mlir::RankedTensorType>(intype);
    if (!inCastType) {
        auto inCastUnrankedType = mlir::dyn_cast<mlir::UnrankedTensorType>(intype);
        outtype = mlir::RankedTensorType::get(shapeData, inCastUnrankedType.getElementType());
    } else {
        outtype = mlir::RankedTensorType::get(shapeData, inCastType.getElementType());
    }

    ON_DEBUG(llvm::errs() << "Got outtype: " << outtype << "\n");

    auto output = builder_.create<mlir::tosa::ReshapeOp>(
        builder_.getUnknownLoc(),
        outtype,
        *input,
        shapeDense
    );

    ON_DEBUG(llvm::errs() << "Got output:" << output << "\n");
    setSSA(node.output, output);
    MSG("VISITED RESHAPE\n");
}

void MLIRPrinter::Visit(const Ops::OpSigmoid& node) {
    MSG("SIGMOID\n");
    MSG("VISITED SIGMOID\n");
}

void MLIRPrinter::Visit(const Ops::OpFlatten& node) {
    MSG("FLATTEN\n");
    MSG("VISITED FLATTEN\n");
}

void MLIRPrinter::Visit(const Core::BebraTensor& tensor) {
    MSG("TENSOR\n");
    auto name = tensor.getName();
    LOG("got tensor name {}\n", name);
    auto ssa = getSSA(name);

    if (!ssa) {
        Core::BebraWarn("SSA not set for tensor " + name);
        mlir::RankedTensorType ttype;
        if (auto knownType = getType(name)) {
            if (auto castedType = mlir::dyn_cast<mlir::RankedTensorType>(*knownType)) {
                ttype = castedType;
            }
        } else {
            ttype = createTensorType(tensor);
        }

        auto& data = tensor.data();
        LOG("Data size for tensor {} is {}\n", name, data.size());

        auto denseAttr = mlir::DenseElementsAttr::get(ttype, llvm::ArrayRef(data));
        LOG("denseAttr size = {}\n", denseAttr.getRawData().size());

        ON_DEBUG(llvm::errs() << "==================" << "\n");
        ON_DEBUG(llvm::errs() << "creating ConstOp"   << "\n");
        ON_DEBUG(llvm::errs() << "type: " << ttype    << "\n");
        ON_DEBUG(llvm::errs() << denseAttr.getType() <<  "\n");
        ON_DEBUG(llvm::errs() << "==================" << "\n");


        mlir::Value ssa_val = builder_.create<mlir::tosa::ConstOp>(builder_.getUnknownLoc(), ttype, denseAttr);
        ON_DEBUG(llvm::errs() << "Type" << ssa_val.getType() << "\n");
        setSSA(name, ssa_val);
        MSG("VISITED TENSOR\n");
        return;
    }

    MSG("VISITED TENSOR\n");
    return;
}

// ====================================================================================

MLIRPrinter::MLIRPrinter(Core::BebraGraph& graph) : builder_(&context_) {
    for (auto&& tensor : graph.tensor_map_) {
        auto&& tname = tensor.first;
        auto&& thetensor = tensor.second;
        auto&& type = createTensorType(thetensor);
        setType(tname, type);
    }
}

void MLIRPrinter::dump(const std::string& filename, const std::string& dumped) {
    std::ofstream os(filename);
    os << dumped;
}

void MLIRPrinter::generate(const Core::BebraGraph& graph, std::string& out_str) {
    // loading needed dialects
    // e.g: tosa, linalg, etc.
    loadAllNeededDialects();

    // Register the translation interfaces


// Inside your initialization:
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    mlir::registerConvertElementwiseToLinalgPass();
    mlir::registerAllToLLVMIRTranslations(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);


    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
    // getting location
    auto loc = builder_.getUnknownLoc();

    // creating a module (to create a function then...)
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));

    // setting insert point
    builder_.setInsertionPointToStart(module.getBody());

    // then i create a void function, so i can run functional passes
    // that's a temporary solution, of course i need to set
    // inputs as the args and return output
    auto inputTypes  = createInputTypes(graph);
    auto outputTypes = createOutputTypes(graph);

    auto funcType = mlir::FunctionType::get(&context_, inputTypes, outputTypes); // void main()
    auto funcOp = builder_.create<mlir::func::FuncOp>(loc, "main", funcType);
    auto* block = funcOp.addEntryBlock();
    builder_.setInsertionPointToStart(block);

    MSG("Checking types before creating FunctionType...\n");

    for (auto t : inputTypes) {
        if (!t) {
            std::cerr << "CRITICAL: Found null input type!" << std::endl;
            abort();
        }
        ON_DEBUG(t.dump());
    }

    for (auto t : outputTypes) {
        if (!t) {
            std::cerr << "CRITICAL: Found null output type!" << std::endl;
            abort();
        }
        ON_DEBUG(t.dump());
    }

    MSG("Context pointer: " << builder_.getContext());

    llvm::raw_string_ostream ss(out_str);
    mlir::OpPrintingFlags flags;

    // initializing start SSA-values
    // this can be : inputs, weights and other stuff that we
    // need to init a neural network

    // ==================================================================

    for (auto&& initializer_name : graph.initializers_) {
        auto&& initializer = graph.getTensor(initializer_name);
        Visit(initializer);
    }

    for (auto&& input_name : graph.inputs_) {
        MSG("VISITING INPUTS\n");
        auto&& input = graph.getTensor(input_name);
        Visit(input);
    }

    // ==================================================================
    MSG("VISITING NODES\n");
    for (auto&& node : graph.nodes_) {
        std::visit([this](auto&& op) { Visit(op); }, node.op_);
    }

    // collecting return values after we finished traversing
    // graph
    auto returnValues = collectReturnValues(graph);
    builder_.create<mlir::func::ReturnOp>(loc, returnValues);


    auto result = compileToLLVM(module, ss);
    if (mlir::succeeded(result)) {
        Core::BebraGreen("successfully compiled model to llvm!");
    }
}

// ================= mlir-specific ==========================


mlir::LogicalResult MLIRPrinter::compileToLLVM(mlir::ModuleOp module, llvm::raw_string_ostream& stream) {
    MSG("Compiling to LLVM!\n");

    // stream << "===== BEFORE OPT ====" << "\n";

    mlir::OpPrintingFlags flags;
    // module.print(stream, flags);

    context_.disableMultithreading();
    mlir::PassManager pm(&context_);
    pm.enableVerifier();
    pm.enableIRPrinting();

// ============================================================================
// PHASE 1: TOSA-TO-LINALG
// ============================================================================

    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaInferShapesPass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToArith());
    pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
    pm.addPass(mlir::createConvertTensorToLinalgPass());

// ============================================================================
// PHASE 2: BUFFERIZATION (convert tensors to memrefs)
// ============================================================================

    pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
    mlir::bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.allowReturnAllocsFromLoops = true;
    options.allowUnknownOps = true;
    options.copyBeforeWrite = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

// ============================================================================
// PHASE 3: lowering to standard control flow (generic into cycles)
// ============================================================================

    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
    pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    pm.addPass(mlir::createConvertSCFToCFPass());

    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToParallelLoopsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToAffineLoopsPass());
    // pm.addPass(mlir::createConvertLinalgToStandardPass());

// ============================================================================
// PHASE 4: lowering to LLVM IR
// ============================================================================

    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    // stream << "===== AFTER OPT ====" << "\n";


    auto result = pm.run(module);
    if (mlir::failed(result)) {
        ON_DEBUG(llvm::errs() << "Pass pipeline failed!\n");
        module.print(stream, flags);
        return mlir::failure();
    }

    module.print(stream, flags);

//     llvm::LLVMContext llvmContext;
//     std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
//
//     if (!llvmModule) {
//         ON_DEBUG(llvm::errs() << "Failed to create llvmModule" << "\n";);
//         return mlir::failure();
//     }
//
//     llvmModule->print(stream, nullptr);
    return mlir::success();
}

llvm::SmallVector<mlir::Value> MLIRPrinter::collectReturnValues(const Core::BebraGraph& graph) {
    llvm::SmallVector<mlir::Value, 4> returnValues;
        for (auto&& outputName : graph.outputs_) {
            auto val = getSSA(outputName);
            if (!val) {
                Core::BebraErr("Error: output value not found in return values: " + outputName);
            }
            ON_DEBUG(llvm::errs() << "Got output:" << outputName << " : " << *val << "\n");
            returnValues.push_back(*val);
        }
    return returnValues;
}

mlir::RankedTensorType MLIRPrinter::broadCastType(mlir::RankedTensorType type, size_t toRank) {
    llvm::SmallVector<int64_t, 4> newShape(toRank, 1);
    auto curShape = type.getShape();

    auto curSize = curShape.size();
    for (size_t i = 0; i < curSize; ++i) {
        newShape[toRank - curSize + i] = curShape[i];
    }

    auto newType = mlir::RankedTensorType::get(newShape, type.getElementType());
    return newType;
}

void MLIRPrinter::loadAllNeededDialects() {
    context_.loadDialect<mlir::arith::ArithDialect>();
    context_.loadDialect<mlir::tensor::TensorDialect>();
    context_.loadDialect<mlir::func::FuncDialect>();
    context_.loadDialect<mlir::linalg::LinalgDialect>();
    context_.loadDialect<mlir::tosa::TosaDialect>();
}

mlir::DenseI64ArrayAttr MLIRPrinter::getDenseI64ArrayAttrFromValue(mlir::Value value) {
    auto defOp = value.getDefiningOp();
    auto constOp = mlir::dyn_cast<mlir::tosa::ConstOp>(defOp);
    if (!constOp) {
        std::cout << " ALERT! " << "\n";
        value.dump();
        throw Core::BebraErr("can't get dense attr from value");
    }

    mlir::Attribute attr = constOp.getValue();

    if (auto denseAttr = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
        return denseAttr;
    }

    if (auto elementsAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
        if (elementsAttr.getElementType().isInteger(64)) {
            std::vector<int64_t> data(elementsAttr.value_begin<int64_t>(), elementsAttr.value_end<int64_t>());
            return mlir::DenseI64ArrayAttr::get(value.getContext(), data);
        }
    }

    Core::BebraWarn("can't cast to denseI64 in getDenseI64ArrayAttrFromValue");
    return nullptr;
}

llvm::SmallVector<mlir::Type> MLIRPrinter::createInputTypes(const Core::BebraGraph& graph) {
    MSG("Creating input types...\n");
    auto inputs = graph.inputs_;
    auto sz = inputs.size();
    llvm::SmallVector<mlir::Type> typeVec;
    typeVec.reserve(sz);

    for (auto&& input : inputs) {
        LOG("Got input {}\n", input);
        auto tensor = graph.getTensor(input);
        auto type = createTensorType(tensor);

        ON_DEBUG(llvm::errs() << "=========================" << "\n");
        ON_DEBUG(llvm::errs() << "Got input type - " << type << "\n");
        ON_DEBUG(llvm::errs() << "=========================" << "\n");

        setType(input, type);
        typeVec.push_back(type);

    }
    return typeVec;
}

llvm::SmallVector<mlir::Type> MLIRPrinter::createOutputTypes(const Core::BebraGraph& graph) {
    MSG("Creating output types..\n.");
    auto outputs = graph.outputs_;
    auto sz = outputs.size();
    llvm::SmallVector<mlir::Type> typeVec;
    typeVec.reserve(sz);

    for (auto&& output : outputs) {
        LOG("Got output {}\n", output);
        auto tensor = graph.getTensor(output);
        auto type = createTensorType(tensor);

        ON_DEBUG(llvm::errs() << "=========================" << "\n" );
        ON_DEBUG(llvm::errs() << "Got output type - " << type << "\n");
        ON_DEBUG(llvm::errs() << "=========================" << "\n" );

        setType(output, type);
        typeVec.push_back(type);
    }
    return typeVec;
}

mlir::Type MLIRPrinter::getElementType(Core::BebraType type) {
    mlir::Type elementType;
    switch (type) {
        case Core::BebraType::FLOAT:
            elementType = builder_.getF32Type();
            break;
        case Core::BebraType::DOUBLE:
            elementType = builder_.getF64Type();
            break;
        case Core::BebraType::INT32:
            elementType = builder_.getI32Type();
            break;
        case Core::BebraType::INT64:
            elementType = builder_.getI64Type();
            break;
        case Core::BebraType::BOOL:
            elementType = builder_.getI8Type();
            break;
        case Core::BebraType::INT8:
            elementType = builder_.getI8Type();
            break;
        case Core::BebraType::UINT8:
            elementType = builder_.getI8Type();
            break;
        default:
            elementType = builder_.getF32Type();
    }
    return elementType;
}

mlir::RankedTensorType MLIRPrinter::createTensorType(const Core::BebraTensor& tensor) {
    mlir::Type elementType = getElementType(tensor.dtype);
    return mlir::RankedTensorType::get(tensor.getShape(), elementType);
}

mlir::RankedTensorType MLIRPrinter::createDynamicTensorType(mlir::Value& tensor, size_t ndims) {
    auto ttype = mlir::dyn_cast<mlir::RankedTensorType>(tensor.getType());
    if (!ttype) {
        throw Core::BebraErr("Value is not a RankedTensor while creating dynamic tensor...");
    }
    auto eltype = ttype.getElementType();
    llvm::SmallVector<int64_t> shape(ndims, mlir::ShapedType::kDynamic);
    return mlir::RankedTensorType::get(shape, eltype);
}

mlir::UnrankedTensorType MLIRPrinter::createUnrankedTensorType(mlir::Type eltype) {
    return mlir::UnrankedTensorType::get(eltype);
}

mlir::UnrankedTensorType MLIRPrinter::createUnrankedTensorType(mlir::Value& tensor) {
    auto type = tensor.getType();
    if (auto casted = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
        return mlir::UnrankedTensorType::get(casted.getElementType());
    } else if (auto castedUnranked = mlir::dyn_cast<mlir::UnrankedTensorType>(type)) {
        return mlir::UnrankedTensorType::get(castedUnranked.getElementType());
    }

    return mlir::UnrankedTensorType::get(builder_.getI64Type());
}
mlir::RankedTensorType MLIRPrinter::createDynamicTensorType(mlir::Value& tensor) {
    auto ttype = mlir::dyn_cast<mlir::RankedTensorType>(tensor.getType());
    if (!ttype) {
        throw Core::BebraErr("Value is not a RankedTensor while creating dynamic tensor...");
    }
    auto ndims = ttype.getRank();
    auto eltype = ttype.getElementType();
    llvm::SmallVector<int64_t> shape(ndims, mlir::ShapedType::kDynamic);
    return mlir::RankedTensorType::get(shape, eltype);
}

mlir::RankedTensorType MLIRPrinter::createDynamicTensorType(size_t ndims, mlir::Type eltype) {
    llvm::SmallVector<int64_t> shape(ndims, mlir::ShapedType::kDynamic);
    return mlir::RankedTensorType::get(shape, eltype);
}
mlir::Value MLIRPrinter::createFilledTensor(mlir::RankedTensorType outtype) {

    auto loc = builder_.getUnknownLoc();
    mlir::ArrayRef<int64_t> outputShape = outtype.getShape();

    // Create output tensor with tensor.empty
    auto emptyTensor = builder_.create<mlir::tensor::EmptyOp>(
        loc, outputShape, outtype.getElementType());

    // Create zero constant for initialization
    auto zero = builder_.create<mlir::arith::ConstantOp>(loc,
        outtype.getElementType(),
        builder_.getZeroAttr(outtype.getElementType()));

    // Fill the output tensor with zeros
    auto filledTensor = builder_.create<mlir::linalg::FillOp>(
        loc, mlir::ValueRange{zero}, mlir::ValueRange{emptyTensor})
                             .getResult(0);

    return filledTensor;
}


mlir::Value MLIRPrinter::convertNCHWToNHWCVal(mlir::Value& val) {
    MSG("converting value form NCHW to NHWC\n");
    llvm::errs() << "got value " << val << "| converting NCHW->NHWC" << "\n";
    auto tname = getNameBySSA(val);
    if (getStoreType(tname) == DataStoreType::NHWC) {
        llvm::errs() << tname << " NHWC type already..." << "\n";
        return val;
    }


    llvm::errs() << tname << " not NHWC type" << "\n";
    auto loc = builder_.getUnknownLoc();
    auto newInputType = convertNCHWToNHWC((val).getType());
    llvm::SmallVector<int32_t, 4> perms{0, 2, 3, 1};
    auto permType = mlir::RankedTensorType::get({4}, builder_.getI32Type());
    auto permAttr = mlir::DenseIntElementsAttr::get(permType, llvm::ArrayRef<int32_t>(perms));
    auto permConstOp = builder_.create<mlir::tosa::ConstOp>(loc, permType, permAttr);
    MSG("creating transpose...\n");
    auto transOp = builder_.create<mlir::tosa::TransposeOp>(
        loc,
        newInputType,
        val,
        permConstOp.getResult()
    );

    llvm::errs() << transOp << "\n";
    setStoreType(tname, DataStoreType::NHWC);
    return transOp;
}

mlir::RankedTensorType MLIRPrinter::convertNCHWToNHWC(mlir::Type type) {
    auto casted = mlir::dyn_cast<mlir::RankedTensorType>(type);
    auto shape = casted.getShape();

    if (shape.size() != 4) {
        Core::BebraErr("You can't convertNCHWToNHWC if your shape size is less then 4...");
    }

    auto eltype = casted.getElementType();

    //                                    N         H          W         C
    llvm::SmallVector<int64_t> outshape{shape[0], shape[2], shape[3], shape[1]};

    return mlir::RankedTensorType::get(outshape, eltype);
}


//NOTE - conv is computed in NHWC type so
// provide input in NHWC not in NCHW
mlir::RankedTensorType MLIRPrinter::computeConv2DOutputType(
    mlir::Value input,
    mlir::Value weight,
    mlir::DenseArrayAttr kernel,
    mlir::DenseArrayAttr stride,
    mlir::DenseArrayAttr pad,
    mlir::DenseArrayAttr dilation,
    mlir::Type accType) {

    MSG("Computing conv2d output type...\n");
    auto inputType =  mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(weight.getType());

    auto kernelValues = kernel.getData();
    auto padValues = pad.getData();
    auto strideValues = stride.getData();
    auto dilationValues = dilation.getData();

    if (!inputType || !weightType) {
      return mlir::RankedTensorType::get({}, accType);
    }

    auto inputShape = inputType.getShape();
    auto weightShape = weightType.getShape();

    int64_t N = inputShape[0];      LOG("N = {}\n", N);
    int64_t C_in = inputShape[3];   LOG("C = {}\n", C_in);
    int64_t H_in = inputShape[1];   LOG("H = {}\n", H_in);
    int64_t W_in = inputShape[2];   LOG("W = {}\n", W_in);

    // TOSA Weights: [OC, KH, KW, IC]
    int64_t C_out = weightShape[0]; LOG("C_out = {}\n", C_out);
    int64_t K_h = weightShape[1];   LOG("K_h = {}\n", K_h);
    int64_t K_w = weightShape[2];   LOG("K_w = {}\n", K_w);
    int64_t IC_weight = weightShape[3]; LOG("IC_w = {}\n", IC_weight);

    if (!K_w) { K_w = K_h; }

    auto getVal = [](mlir::DenseArrayAttr attr, size_t idx, int64_t defaultVal) {
        auto data = attr.getData();
        auto val = (data.size() > idx) ? data[idx] : defaultVal;
        return val != 0 ? val : defaultVal;
    };

    int64_t stride_h = getVal(stride, 0, 1);
    int64_t stride_w = getVal(stride, 1, stride_h);
    int64_t dil_h = getVal(dilation, 0, 1);
    int64_t dil_w = getVal(dilation, 1, 1);

    // TOSA Order: [top, bottom, left, right]
    int64_t p_top = getVal(pad, 0, 0);
    int64_t p_bottom = getVal(pad, 1, 0);
    int64_t p_left = getVal(pad, 2, 0);
    int64_t p_right = getVal(pad, 3, 0);

    int64_t eff_K_h = dil_h * (K_h - 1) + 1;
    int64_t eff_K_w = dil_w * (K_w - 1) + 1;

    auto calculateDim = [](int64_t in, int64_t k, int64_t p_total, int64_t s) -> int64_t {
        if (in == -1) return -1;
        int64_t out = (in + p_total - k) / s + 1;
        return out > 0 ? out : 0;
    };

    // int64_t W_out = calculateDim(W_in, eff_K_w, p_left + p_right, stride_w);
    // int64_t H_out = calculateDim(H_in, eff_K_h, p_top + p_bottom, stride_h);

    int64_t H_out = mlir::ShapedType::kDynamic;
    int64_t W_out = mlir::ShapedType::kDynamic;

    LOG("pads = [{}, {}, {}, {}]\n", p_left, p_right, p_top, p_bottom);

    llvm::SmallVector<int64_t> outputShape = {N, H_out, W_out, C_out};
    auto elementType = inputType.getElementType();

    MSG("Finished computing conv2d output\n");
    return mlir::RankedTensorType::get(outputShape, elementType);
}

mlir::RankedTensorType MLIRPrinter::computeMatMulOutputType(mlir::Value& input_a,
                                                            mlir::Value& input_b) {
    MSG("Computing matmul output type\n");

    ON_DEBUG(llvm::errs() << "input_a = " << input_a << "\n");
    ON_DEBUG(llvm::errs() << "input_b = " << input_b << "\n");

    auto typeA = input_a.getType(); ON_DEBUG(llvm::errs() << "Atype = " << typeA << "\n");
    auto typeB = input_b.getType(); ON_DEBUG(llvm::errs() << "Btype = " << typeB << "\n");

    auto inputAType = mlir::dyn_cast<mlir::RankedTensorType>(typeA);
    if (!inputAType) {
        Core::BebraErr("inputA in matmul is nor RankedTensor");
    }
    auto inputBType = mlir::dyn_cast<mlir::RankedTensorType>(typeB);
    if (!inputBType) {
        Core::BebraErr("inputB in matmul is nor RankedTensor");
    }

    ON_DEBUG(llvm::errs() << "casted to rankedTensor..." << "\n");

    auto inputAShape = inputAType.getShape();
    auto inputBShape = inputBType.getShape();

    ON_DEBUG(llvm::errs() << "Got shapes..." << "\n");
    if (inputAShape.size() != 2) {
        Core::BebraErr("input shape in matmul != 2, now 2D vectors are available only!");
    }

    auto H_a = inputAShape[0]; LOG("H_a = {}\n", H_a);
    auto W_a = inputAShape[1]; LOG("W_A = {}\n", W_a);

    auto H_b = inputBShape[0]; LOG("H_b = {}\n", H_b);
    auto W_b = inputBShape[1]; LOG("W_b = {}\n", W_b);

    llvm::SmallVector<int64_t> outputShape = {H_a, W_b};
    auto eltype = inputAType.getElementType();

    MSG("returning tensor type...\n");
    return mlir::RankedTensorType::get(outputShape, eltype);
}

mlir::RankedTensorType MLIRPrinter::computeMaxPool2DOutputType(
    mlir::Value input,
    mlir::DenseArrayAttr kernel_size,
    mlir::DenseArrayAttr stride,
    mlir::DenseArrayAttr pad,
    mlir::DenseArrayAttr dilation) {

    MSG("Computing maxpool2d output type...\n");

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType) {
        return mlir::RankedTensorType::get({-1, -1, -1, -1}, mlir::Float32Type::get(input.getContext()));
    }

    auto inputShape = inputType.getShape();

    // NHWC Layout
    int64_t N = inputShape[0];
    int64_t H_in = inputShape[1];
    int64_t W_in = inputShape[2];
    int64_t C_in = inputShape[3];

    // Лямбда для безопасного извлечения атрибутов
    auto getSafeVal = [](mlir::DenseArrayAttr attr, size_t idx, int64_t defaultVal) {
        if (!attr || attr.getData().size() <= idx) return defaultVal;
        int64_t val = attr.getData()[idx];
        return val > 0 ? val : defaultVal;
    };

    int64_t K_h = getSafeVal(kernel_size, 0, 1);
    int64_t K_w = getSafeVal(kernel_size, 1, K_h);

    int64_t stride_h = getSafeVal(stride, 0, 1);
    int64_t stride_w = getSafeVal(stride, 1, stride_h);

    int64_t dil_h = getSafeVal(dilation, 0, 1);
    int64_t dil_w = getSafeVal(dilation, 1, 1);

    // TOSA Order: [top, bottom, left, right]
    int64_t p_t = getSafeVal(pad, 0, 0);
    int64_t p_b = getSafeVal(pad, 1, 0);
    int64_t p_l = getSafeVal(pad, 2, 0);
    int64_t p_r = getSafeVal(pad, 3, 0);

    int64_t eff_K_h = dil_h * (K_h - 1) + 1;
    int64_t eff_K_w = dil_w * (K_w - 1) + 1;

    auto calcDim = [](int64_t in, int64_t k, int64_t p, int64_t s) -> int64_t {
        if (in == mlir::ShapedType::kDynamic) return mlir::ShapedType::kDynamic;
        int64_t out = (in + p - k) / s + 1;
        return out > 0 ? out : 0;
    };

    // int64_t H_out = calcDim(H_in, eff_K_h, p_t + p_b, stride_h);
    // int64_t W_out = calcDim(W_in, eff_K_w, p_l + p_r, stride_w);


    int64_t H_out = mlir::ShapedType::kDynamic;
    int64_t W_out = mlir::ShapedType::kDynamic;

    int64_t C_out = C_in;
    LOG("MaxPool2D: In[{}x{}x{}x{}] -> Out[{}x{}x{}x{}] (K={}x{}, S={}x{}, P=[{},{},{},{}])\n",
        N, H_in, W_in, C_in, N, H_out, W_out, C_out, K_h, K_w, stride_h, stride_w, p_t, p_b, p_l, p_r);

    llvm::SmallVector<int64_t> outputShape = {N, H_out, W_out, C_out};
    return mlir::RankedTensorType::get(outputShape, inputType.getElementType());
}
} // namespace Bebra::MLIR
