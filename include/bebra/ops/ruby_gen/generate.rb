require_relative "count_output.rb"

def generate_header
<<-HEADER
#pragma once
#include <string>
namespace Bebra { namespace Core { class BebraGraph; } }
#include <vector>
#include <iostream>
#include "bebra/core/BebraErr.hpp"
#include "bebra/core/BebraColors.hpp"
#include "bebra/ops/BebraVisitor.hpp"
#include "bebra/ops/CountShapeHelpers.hpp"
namespace Bebra {
namespace Ops {
HEADER
end

def generate_factory_hpp_header
<<-HEADER
#pragma once
#include "bebra/core/BebraNode.hpp"
#include "bebra/core/BebraAttr.hpp"
#include "bebra/ops/BebraOperators.hpp"

namespace Bebra::Ops {

OpVariant CreateOp(const std::string& op_type,
                   const onnx::NodeProto& onnx_node,
                   const std::vector<std::string>& inputs,
                   const std::vector<std::string>& outputs,
                   const std::unordered_map<std::string, Core::Attr>& attrs);
HEADER
end

def generate_verify_header()
<<-HEADER
#include "bebra/ops/BebraOperators.hpp"
#include "bebra/core/BebraGraph.hpp"
#include "bebra/ops/CountShapeHelpers.hpp"
namespace Bebra::Ops {
HEADER
end

def generate_verify_footer()
<<-FOOTER
}
FOOTER
end

def generate_factory_hpp_footer()
<<-FOOTER
}
FOOTER
end

def generate_if(op)
    name = op["name"]
<<-CPP
    if (op_type == "#{name}") {
        return CreateOp#{name}(onnx_node, inputs, outputs, attrs);
    }
CPP
end

def generate_factory_cpp_header(ops)
    op_ifs = ops.map{|op| generate_if(op)}
<<-HEADER
#include "bebra/ops/BebraFactory.hpp"
namespace Bebra::Ops {

OpVariant CreateOp(const std::string& op_type,
                   const onnx::NodeProto& onnx_node,
                   const std::vector<std::string>& inputs,
                   const std::vector<std::string>& outputs,
                   const std::unordered_map<std::string, Core::Attr>& attrs)

{
    #{op_ifs.join()}

    return CreateOpVoid(onnx_node, inputs, outputs, attrs);
}
HEADER
end


def generate_factory_cpp_footer
<<-FOOTER
}
FOOTER

end





def generate_variant(ops)
<<~CPP
#pragma once
namespace Bebra::Ops {
// OpVariant for every node type
using OpVariant = std::variant<#{(ops.map {|op| "Op#{op["name"]}"}).join(",")}>;
}
CPP
end

def generate_footer
    <<~FOOTER
    } // end of Ops :0
    } // end of Bebra :0
    FOOTER
end

def generate_attr(attr, op_name)
    name = attr["name"]
    type = YAML_TO_CPP_TYPE_HASH[attr["type"]]
    if (attr.key?("default"))
        formatted_val = format_value(attr["default"])
    <<-CPP
    #{type} #{name} = #{type}(#{formatted_val});
    CPP
    else
    <<-CPP
    #{type} #{name};
    CPP
    end
end

def generate_input(input, name)
    name = input["name"]
    <<-CPP
        std::string #{name};
    CPP
end

def generate_output(output, name)
    name = output["name"]
    <<-CPP
        std::string #{name};
    CPP

end

def generate_shape_verify(shape, name)
    cpp = "bool Op#{name}::verify(const Core::BebraGraph& graph) const {\n"
    template = VERIFY_TEMPLATES[shape.to_sym]
    if template.nil?
        puts "Nil verify shape template for " + name + " " + shape
        return;
    end
    cpp << template.call(name)
    cpp << "}\n"
    cpp
end

def generate_count_output_shape(shape, name)
    cpp = "std::vector<int64_t> Op#{name}::countOutputShape(Core::BebraGraph& graph) {"

    template = COUNT_SHAPE_TEMPLATES[shape.to_sym]
    if template.nil?
        puts "Nil count shape template for " + name + " category:" + shape
        return;
    end
    cpp << template.call(name)
    cpp << "}\n"
    cpp
end

def generate_op(op)

    name = op["name"]
    attrs = []
    shape_verify = []
    inputs = []
    outputs = []
    attrs_line = []

    if op["inputs"]
        inputs = op["inputs"].map { |input| generate_input(input, name)}
    end

    if op["outputs"]
        outputs = op["outputs"].map { |output| generate_input(output, name)}
    end

    if op["attributes"]
        attrs = op["attributes"].map { |attr| generate_attr(attr, name) }
        attrs_line = (op["attributes"].map {|op| "\"#{op["name"]}\""}).join(",\n")
    end

    if not op["attributes"]
        attrs_line = ""
    end

    <<~CPP
    struct Op#{name} {
    void accept(BebraVisitor& visitor) { visitor.Visit(*this); }
    void accept(const BebraVisitor& visitor) { visitor.Visit(*this); }
    static constexpr const char* getOpType() { return "#{name}"; }
    std::vector<int64_t> countOutputShape(Core::BebraGraph& graph);
    const std::vector<std::string> getAttrsString() const {
        return {
            #{attrs_line}
        };
    }

    #{inputs.join()}
    #{outputs.join()}
    #{attrs.join()}
    bool verify(const Core::BebraGraph& graph) const;
        };
    CPP

end

def generate_op_creator_hpp(op)
<<-CPP
OpVariant CreateOp#{op["name"]} (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs);
CPP
end

def generate_inputs(cpp, inputs, name)
    required_count = inputs.count { |i| i["required"] }
    if required_count > 0
        cpp << "        if (inputs.size() < #{required_count}) {\n"
        cpp << "            throw Core::BebraErr(\"#{name}: expected #{required_count} required input(s)\");\n"
        cpp << "        }\n"
    end

    inputs.each_with_index do |inp, idx|
        if inp["required"]
            cpp << "        op.#{inp["name"]} = inputs[#{idx}];\n"
        else
            cpp << "        if (inputs.size() > #{idx} && !inputs[#{idx}].empty()) op.#{inp["name"]} = inputs[#{idx}];\n"
        end
    end
    cpp << "\n"
end

def generate_outputs(cpp, outputs, name)
    required_count = outputs.count { |o| o["required"] }
    if required_count > 0
        cpp << "        if (outputs.size() < #{required_count}) {\n"
        cpp << "            throw Core::BebraErr(\"#{name}: expected #{required_count} required output(s)\");\n"
        cpp << "        }\n"
    end

    outputs.each_with_index do |out, idx|
        if out["required"]
            cpp << "        op.#{out["name"]} = outputs[#{idx}];\n"
        else
            cpp << "        if (outputs.size() > #{idx}) op.#{out["name"]} = outputs[#{idx}];\n"
        end
    end
    cpp << "\n"
end

def get_cpp_type(yaml_type, type_hash)
    key = yaml_type.to_s.strip
    result = type_hash[key]
    return result ? result.to_s : "AttrVal"
end

def format_default_value(val, cpp_type_str)
    return "nullptr" if val.nil?

    if cpp_type_str.include?("std::vector") && val.is_a?(Array)
      return "{#{val.join(", ")}}"
    end

    if cpp_type_str == "std::string"
      val_str = val.to_s
      return val_str.start_with?('"') ? val_str : "\"#{val_str}\""
    end

    return val.to_s
  end

def generate_attributes(cpp, attributes, name)
    attributes.select { |a| a["required"] }.each do |attr|
        attr_name = attr["name"]
        attr_type = get_cpp_type(attr["type"], YAML_TO_CPP_TYPE_HASH)
        cpp << "        {\n"
        cpp << "            auto it = attrs.find(\"#{attr_name}\");\n"
        cpp << "            if (it == attrs.end()) throw Core::BebraErr(\"#{name}: missing required attr '#{attr_name}'\");\n"
        cpp << "            op.#{attr_name} = it->second.getValRef<#{attr_type}>();\n"
        cpp << "        }\n"
    end

    attributes.reject { |a| a["required"] }.each do |attr|
        attr_name = attr["name"]
        attr_type = get_cpp_type(attr["type"], YAML_TO_CPP_TYPE_HASH)
        default = format_default_value(attr["default"], attr_type)
        cpp << "        {\n"
        cpp << "            auto it = attrs.find(\"#{attr_name}\");\n"
        cpp << "            op.#{attr_name} = (it != attrs.end())\n"
        cpp << "                ? it->second.getValRef<#{attr_type}>()\n"
        cpp << "                : #{attr_type}(#{default});\n"
        cpp << "        }\n"
    end
end

def generate_op_creator_cpp(op)
name = op["name"]
cpp = <<-CPP
OpVariant CreateOp#{name} (const onnx::NodeProto& node,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::unordered_map<std::string, Core::Attr>& attrs) {
    Op#{name} op;
CPP

if op["inputs"]&.any?
    generate_inputs(cpp, op["inputs"], name)
end
if op["outputs"]&.any?
    generate_outputs(cpp, op["outputs"], name)
end

if op["attributes"]&.any?
    generate_attributes(cpp, op["attributes"], name)
end
cpp << "\treturn op;\n}\n"
cpp
end

def generate_visitor_class(ops)
    hpp = <<-CPP
        #pragma once
        namespace Bebra::Ops {
    CPP

    ops.each do |op|
    hpp << "struct Op#{op["name"]};\n"
    end

    hpp << "class BebraVisitor {\npublic:\n"
    ops.each do |op|
    hpp << "\tvirtual void Visit(Op#{op["name"]}& node) = 0;\n"
    hpp << "\tvirtual void Visit(const Op#{op["name"]}& node) const = 0;\n"
    end
    hpp << "\tvirtual ~BebraVisitor() = default;\n"
    hpp << "};\n"

    hpp << "}\n"
    return hpp

end
