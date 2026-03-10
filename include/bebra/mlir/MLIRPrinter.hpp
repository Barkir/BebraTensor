#pragma once
#include "bebra/core/BebraErr.hpp"
#include "bebra/ops/BebraOperators.hpp"

namespace Bebra::Core { class BebraGraph; } // forward declaration

namespace Bebra::MLIR {

class MLIRPrinter {


public: // constructor
	MLIRPrinter(const Core::BebraGraph& graph) {
		generate(graph);
	}

	MLIRPrinter() = default;
	~MLIRPrinter() = default;


public: // helpers
	std::string generate(const Core::BebraGraph& graph);

public: // Visitors
	void Visit(const Ops::OpVoid& node) const;
	void Visit(const Ops::OpConv& node) const;
	void Visit(const Ops::OpGemm& node) const;
	void Visit(const Ops::OpAdd& node) const;
	void Visit(const Ops::OpRelu& node) const;
	void Visit(const Ops::OpMul& node) const;
	void Visit(const Ops::OpMatMul& node) const;
	void Visit(const Ops::OpMaxPool& node) const;
	void Visit(const Ops::OpReduceMean& node) const;
	void Visit(const Ops::OpReshape& node) const;
	void Visit(const Ops::OpSigmoid& node) const;
	void Visit(const Ops::OpFlatten& node) const;

};



}
