        #pragma once
        namespace Bebra::Ops {
struct OpVoid;
struct OpConv;
struct OpGemm;
struct OpAdd;
struct OpRelu;
struct OpMul;
struct OpMatMul;
struct OpMaxPool;
struct OpReduceMean;
struct OpReshape;
struct OpSigmoid;
struct OpFlatten;
class BebraVisitor {
public:
	virtual void Visit(OpVoid& node) = 0;
	virtual void Visit(const OpVoid& node) const = 0;
	virtual void Visit(OpConv& node) = 0;
	virtual void Visit(const OpConv& node) const = 0;
	virtual void Visit(OpGemm& node) = 0;
	virtual void Visit(const OpGemm& node) const = 0;
	virtual void Visit(OpAdd& node) = 0;
	virtual void Visit(const OpAdd& node) const = 0;
	virtual void Visit(OpRelu& node) = 0;
	virtual void Visit(const OpRelu& node) const = 0;
	virtual void Visit(OpMul& node) = 0;
	virtual void Visit(const OpMul& node) const = 0;
	virtual void Visit(OpMatMul& node) = 0;
	virtual void Visit(const OpMatMul& node) const = 0;
	virtual void Visit(OpMaxPool& node) = 0;
	virtual void Visit(const OpMaxPool& node) const = 0;
	virtual void Visit(OpReduceMean& node) = 0;
	virtual void Visit(const OpReduceMean& node) const = 0;
	virtual void Visit(OpReshape& node) = 0;
	virtual void Visit(const OpReshape& node) const = 0;
	virtual void Visit(OpSigmoid& node) = 0;
	virtual void Visit(const OpSigmoid& node) const = 0;
	virtual void Visit(OpFlatten& node) = 0;
	virtual void Visit(const OpFlatten& node) const = 0;
	virtual ~BebraVisitor() = default;
};
}
