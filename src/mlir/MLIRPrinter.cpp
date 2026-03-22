#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

#include "bebra/core/BebraColors.hpp"
#include "bebra/core/BebraErr.hpp"
#include "bebra/core/BebraGraph.hpp"
#include "bebra/core/BebraTensor.hpp"
#include "bebra/core/BebraType.hpp"
#include "bebra/mlir/MLIRPrinter.hpp"
#include "bebra/core/BebraLog.hpp"

#include <iostream>

namespace Bebra::MLIR {

void MLIRPrinter::Visit(const Ops::OpVoid& node) {
    MSG("VOID\n");

    MSG("VISITED VOID\n");
}

void MLIRPrinter::Visit(const Ops::OpConv& node) {
    MSG("CONV\n");

    // inputs
    auto input = getSSA(node.input);
    if (!input) {
        Core::BebraWarn("no input provided + " + node.input);
    }

    auto bias = getSSA(node.bias);
    if (!bias) {
        Core::BebraWarn("no bias provided + " + node.bias);
    }
    auto weight = getSSA(node.weight);
    if (!weight) {
        Core::BebraWarn("no weight provided + " + node.weight);
    }

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
    std::cout << "got tensor attrs for strides and dilations for builder\n";

    auto intype = mlir::dyn_cast<mlir::RankedTensorType>((*input).getType());
    auto eltype = intype.getElementType();

    // auto outtype = createDynamicTensorType(*input);
    auto outtype = computeConv2DOutputType(*input, *weight, kernelShapeAttr,
                                            stridesDenseAttr, padsDenseAttr,
                                            dilationsDenseAttr, eltype);

    auto accTypeAttr = mlir::TypeAttr::get(eltype);

    if (!bias) {
        auto zeroAttr = builder_.getZeroAttr(eltype);
        auto biasType = createDynamicTensorType(1, eltype);
        auto denseAttr = mlir::DenseElementsAttr::get(biasType, zeroAttr);

        bias = builder_.create<mlir::tosa::ConstOp>(
            builder_.getUnknownLoc(),
            biasType,
            denseAttr
        ).getResult();
    }

    Core::BebraWarn("==================");
    llvm::outs() << "input_type = " << (*input).getType() << "\n";
    llvm::outs() << "weights_type =" << (*weight).getType() << "\n";
    llvm::outs() << "bias_type =" << (*bias).getType() << "\n";
    llvm::outs() <<  "strides_type" << stridesDenseAttr.getElementType() << "\n";
    llvm::outs() <<  "dilations_type" << dilationsDenseAttr.getElementType() << "\n";
    llvm::outs() <<  "pads_type = " << padsDenseAttr.getElementType() << "\n";
    llvm::outs() <<  "outtype = " << outtype << "\n";
    Core::BebraWarn("==================");

    auto output = builder_.create<mlir::tosa::Conv2DOp>(builder_.getUnknownLoc(),
                                                                  outtype,
                                                                  *input,
                                                                  *weight,
                                                                  *bias,
                                                                  padsDenseAttr,
                                                                  stridesDenseAttr,
                                                                  dilationsDenseAttr,
                                                                  accTypeAttr
    );
    std::cout << "created output\n";

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

    Core::BebraWarn("==================");
    llvm::outs() << "lhs_type = " << ltype << "\n";
    llvm::outs() << "rhs_type = " << processedRhs.getType() << "\n";
    llvm::outs() << "output_type = " << ltype << "\n";
    Core::BebraWarn("==================");

    output = builder_.create<mlir::tosa::AddOp>(builder_.getUnknownLoc(),
                                                            ltype, *lhs, processedRhs);
    setSSA(node.output, output);
    MSG("VISITED ADD\n");
    return;
}


void MLIRPrinter::Visit(const Ops::OpRelu& node) {
    MSG("RELU\n");

    auto input = getSSA(node.input);
    if (!input) {
        Core::BebraWarn("no input at relu: " + node.input);
    }

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

    // auto intype_a = mlir::dyn_cast<mlir::RankedTensorType>((*input_a).getType());
    // if (intype_a.getRank() != 3) {
    //     LOG("Turning into a 3d tensor {}\n", node.input_a);
    //     auto newType = broadCastType(intype_a, 3);
    //     processedA = builder_.create<mlir::tosa::ReshapeOp>(
    //         builder_.getUnknownLoc(),
    //         newType,
    //         *input_a,
    //         builder_.getDenseI64ArrayAttr(newType.getShape())
    //     ).getResult();
    // }

    // auto intype_b = mlir::dyn_cast<mlir::RankedTensorType>((*input_b).getType());
    // if (intype_b.getRank() != 3) {
    //     LOG("Turning into a 3d tensor {}", node.input_a);
    //     auto newType = broadCastType(intype_b, 3);
    //     processedB = builder_.create<mlir::tosa::ReshapeOp>(
    //         builder_.getUnknownLoc(),
    //         newType,
    //         *input_b,
    //         builder_.getDenseI64ArrayAttr(newType.getShape())
    //     ).getResult();
    // }

    auto outtype = computeMatMulOutputType(processedA, processedB);
    auto filledTensor = createFilledTensor(outtype);

    llvm::errs() << "output = " << outtype << "\n";
    // llvm::errs() << "filledTensor = " << filledTensor << "\n";

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

    std::cout << "creating dilations attrs..." << "\n";
    auto kernelAttr = builder_.getDenseI64ArrayAttr(kernel_shape);
    auto dilationsAttr = builder_.getDenseI64ArrayAttr(dilations);
    auto stridesAttr = builder_.getDenseI64ArrayAttr(strides);
    auto padsAttr = builder_.getDenseI64ArrayAttr(pads);

    auto outtype = computeMaxPool2DOutputType(
        *input,
        kernelAttr,
        stridesAttr,
        padsAttr,
        dilationsAttr
    );
    // auto outtype = createDynamicTensorType(*input);
    // output
    std::cout << "creating output" << std::endl;
    auto output = builder_.create<mlir::tosa::MaxPool2dOp>(
        builder_.getUnknownLoc(),
        outtype,
        *input,
        kernelAttr,
        stridesAttr,
        padsAttr
    );

    setSSA(node.output, output);
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

    llvm::errs() << "input = " << *input << "\n";
    llvm::errs() << "shape = " << *shape << "\n";

    // auto castedShape = mlir::dyn_cast<mlir::tosa::ConstOp>(shape);
    // if (!castedShape) {
        // Core::BebraErr("shape input is not defined in Reshape call! -> " + node.shape);
    // }
    // auto attr = castedShape.getValue();

    auto shapeDense = getDenseI64ArrayAttrFromValue((*shape));
    llvm::errs() << "Got dense array of shape: " << shapeDense << "\n";

    llvm::ArrayRef<int64_t> shapeData = shapeDense.asArrayRef();
    mlir::RankedTensorType outtype;

    auto intype = (*input).getType();
    llvm::errs() << "Got input type: " << intype << "\n";

    auto inCastType = mlir::dyn_cast<mlir::RankedTensorType>(intype);
    if (!inCastType) {
        auto inCastUnrankedType = mlir::dyn_cast<mlir::UnrankedTensorType>(intype);
        outtype = mlir::RankedTensorType::get(shapeData, inCastUnrankedType.getElementType());
    } else {
        outtype = mlir::RankedTensorType::get(shapeData, inCastType.getElementType());
    }

    llvm::errs() << "Got outtype: " << outtype << "\n";

    auto output = builder_.create<mlir::tosa::ReshapeOp>(
        builder_.getUnknownLoc(),
        outtype,
        *input,
        shapeDense
    );

    llvm::errs() << "Got output:" << output << "\n";
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
    std::cout << "got tensor name " << name << "\n";
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

        auto denseAttr = mlir::DenseElementsAttr::get(ttype, llvm::ArrayRef(data));

        mlir::Value ssa_val = builder_.create<mlir::tosa::ConstOp>(builder_.getUnknownLoc(), ttype, denseAttr);
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
        type_map_[tname] = type;
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

    std::cout << "Checking types before creating FunctionType..." << std::endl;

    for (auto t : inputTypes) {
        if (!t) {
            std::cerr << "CRITICAL: Found null input type!" << std::endl;
            abort();
        }
        t.dump();
    }

    for (auto t : outputTypes) {
        if (!t) {
            std::cerr << "CRITICAL: Found null output type!" << std::endl;
            abort();
        }
        t.dump();
    }

    std::cout << "Context pointer: " << builder_.getContext() << std::endl;

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
        std::cout << "VISITING INPUTS" << "\n";
        auto&& input = graph.getTensor(input_name);
        Visit(input);
    }

    // ==================================================================
    std::cout << "VISITING NODES" << "\n";
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

    stream << "===== BEFORE OPT ====" << "\n";

    mlir::OpPrintingFlags flags;
    module.print(stream, flags);

    mlir::PassManager pm(&context_);
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaInferShapesPass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaOptionalDecompositions());
    pm.addPass(mlir::createCSEPass());

    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToArith());

    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());


    stream << "===== AFTER OPT ====" << "\n";
    auto result = pm.run(module);
    if (mlir::failed(result)) {
        llvm::errs() << "Pass pipeline failed!\n";
        module.print(stream, flags);
        return mlir::failure();
    }

    module.print(stream, flags);

    return mlir::success();
}

llvm::SmallVector<mlir::Value> MLIRPrinter::collectReturnValues(const Core::BebraGraph& graph) {
    llvm::SmallVector<mlir::Value, 4> returnValues;
        for (auto&& outputName : graph.outputs_) {
            auto val = getSSA(outputName);
            if (!val) {
                Core::BebraErr("Error: output value not found in return values: " + outputName);
            }
            llvm::errs() << "Got output:" << outputName << " : " << *val << "\n";
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

        llvm::errs() << "=========================" << "\n";
        llvm::errs() << "Got input type - " << type << "\n";
        llvm::errs() << "=========================" << "\n";

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

        llvm::errs() << "=========================" << "\n";
        llvm::errs() << "Got output type - " << type << "\n";
        llvm::errs() << "=========================" << "\n";

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
    int64_t C_in = inputShape[1];   LOG("C = {}\n", C_in);
    int64_t H_in = inputShape[2];   LOG("H = {}\n", H_in);
    int64_t W_in = inputShape[3];   LOG("W = {}\n", W_in);

    int64_t C_out = weightShape[0]; LOG("C_out = {}\n", C_out);
    int64_t K_h = kernelValues[0];   LOG("K_h = {}\n", K_h);
    int64_t K_w = kernelValues[1];   LOG("K_w = {}\n", K_w);
    if (!K_w) { K_w = K_h; }

    auto strSz = strideValues.size();
    int64_t stride_h = strSz > 0 ? strideValues[0] : 1; LOG("stride_h = {}\n", stride_h);
    if (!stride_h) { stride_h = 1; }
    int64_t stride_w = strSz > 1 ? strideValues[1] : 1; LOG("stride_w = {}\n", stride_w);
    if (!stride_w) { stride_w = stride_h; }


    auto dilationSz = dilationValues.size();
    int64_t dilation_h = dilationSz > 0 ? dilationValues[0] : 1; LOG("dilation_h = {}\n", dilation_h);
    int64_t dilation_w = dilationSz > 1 ? dilationValues[1] : 1; LOG("dilation_ц = {}\n", dilation_w);

    auto padSz = padValues.size();
    int64_t pad_left =   padSz > 0 ? padValues[0] : 0;
    int64_t pad_right =  padSz > 1 ? padValues[1] : 0;
    int64_t pad_top =    padSz > 2 ? padValues[2] : 0;
    int64_t pad_bottom = padSz > 3 ? padValues[3] : 0;

    // int64_t H_out = (H_in - K_h + (pad_top + pad_bottom)) / stride_h + 1;
    // int64_t W_out = (W_in - K_w +  (pad_right + pad_left)) / stride_w + 1;

    auto H_out = mlir::ShapedType::kDynamic;
    auto W_out = mlir::ShapedType::kDynamic;

    llvm::SmallVector<int64_t> outputShape = {N, C_out, H_out, W_out};
    auto elementType = inputType.getElementType();

    MSG("Finished computing conv2d output\n");
    return mlir::RankedTensorType::get(outputShape, elementType);
}

mlir::RankedTensorType MLIRPrinter::computeMatMulOutputType(mlir::Value& input_a,
                                                            mlir::Value& input_b) {
    MSG("Computing matmul output type\n");

    llvm::errs() << "input_a = " << input_a << "\n";
    llvm::errs() << "input_b = " << input_b << "\n";

    auto typeA = input_a.getType(); llvm::errs() << "Atype = " << typeA << "\n";
    auto typeB = input_b.getType(); llvm::errs() << "Btype = " << typeB << "\n";

    auto inputAType = mlir::dyn_cast<mlir::RankedTensorType>(typeA);
    if (!inputAType) {
        Core::BebraErr("inputA in matmul is nor RankedTensor");
    }
    auto inputBType = mlir::dyn_cast<mlir::RankedTensorType>(typeB);
    if (!inputBType) {
        Core::BebraErr("inputB in matmul is nor RankedTensor");
    }

    llvm::errs() << "casted to rankedTensor..." << "\n";

    auto inputAShape = inputAType.getShape();
    auto inputBShape = inputBType.getShape();

    llvm::errs() << "Got shapes..." << "\n";
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
    llvm::errs() << "Got input type: " << inputType << "\n";

    if (!inputType) {
        MSG("Error: Input type is not RankedTensorType\n");
        return mlir::RankedTensorType::get({}, inputType ? inputType.getElementType() :
               mlir::Float32Type::get(input.getContext()));
    }

    auto inputShape = inputType.getShape();
    auto kernelValues = kernel_size.getData();
    auto padValues = pad.getData();
    auto strideValues = stride.getData();
    auto dilationValues = dilation.getData();

    int64_t N = inputShape[0];      LOG("N = {}\n", N);
    int64_t C_in = inputShape[1];   LOG("C = {}\n", C_in);
    int64_t H_in = inputShape[2];   LOG("H = {}\n", H_in);
    int64_t W_in = inputShape[3];   LOG("W = {}\n", W_in);

    auto kernelSz = kernelValues.size();
    int64_t K_h = kernelSz > 0 ? kernelValues[0] : 1; LOG("K_h = {}\n", K_h);
    int64_t K_w = kernelSz > 1 ? kernelValues[1] : 1; LOG("K_w = {}\n", K_w);
    if (!K_w) { K_w = K_h; }

    auto strSz = strideValues.size();
    int64_t stride_h = strSz > 0 ? strideValues[0] : 1;
    if (!stride_h) { stride_h = 1; } LOG("stride_h = {}\n", stride_h);

    int64_t stride_w = strSz > 1 ? strideValues[1] : 1;
    if (!stride_w) { stride_w = stride_h; } LOG("stride_w = {}\n", stride_w);

    auto dilationSz = dilationValues.size();
    int64_t dilation_h = dilationSz > 0 ? dilationValues[0] : 1; LOG("dilation_h = {}\n", dilation_h);
    int64_t dilation_w = dilationSz > 1 ? dilationValues[1] : 1; LOG("dilation_w = {}\n", dilation_w);

    auto padSz = padValues.size();
    int64_t pad_left =   padSz > 0 ? padValues[0] : 0;
    int64_t pad_right =  padSz > 1 ? padValues[1] : 0;
    int64_t pad_top =    padSz > 2 ? padValues[2] : 0;
    int64_t pad_bottom = padSz > 3 ? padValues[3] : 0;

    // int64_t H_out = (H_in - K_h + 2 *(pad_top + pad_bottom)) / stride_h + 1;
    // int64_t W_out = ((W_in - K_w + 2 *(pad_right + pad_left)) / stride_w + 1);

    int64_t C_out = C_in;

    /* if (H_out < 0) */ auto H_out = mlir::ShapedType::kDynamic;
    /* if (W_out < 0) */ auto W_out = mlir::ShapedType::kDynamic;

    llvm::SmallVector<int64_t> outputShape = {N, C_out, H_out, W_out};
    auto elementType = inputType.getElementType();

    MSG("Finished computing maxpool2d output\n");
    return mlir::RankedTensorType::get(outputShape, elementType);
}

} // namespace Bebra::MLIR
