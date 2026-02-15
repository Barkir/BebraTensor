YAML_PATH = "ops.yaml"

require 'yaml'

YAML_TO_CPP_TYPE_HASH = {
    "int64[]" => "std::vector<int64_t>",
    "int64" => "int64_t",
    "float32[]" => "std::vector<float>",
    "float32" => "float",
    "string" => "std::string"
}

def format_value(value)
    if value.is_a?(Array)
        "{#{value.join(", ")}}"
    elsif value.is_a?(Float)
        "#{value}f"
    else
        value
    end

end

def generate_header
    <<~HEADER
    #pragma once
    #include <string>
    #include "bebra/core/BebraNode.hpp"
    #include "bebra/core/BebraErr.hpp"
    #include "bebra/core/BebraColors.hpp"
    namespace Bebra {
    namespace Ops {
    HEADER
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
    <<~CPP
        #{type} #{name}() const  {
            auto it = node_->attrs_.find("#{name}");
            if (it == node_->attrs_.end()) {

            std::cout << FG_GOLD_RGB "Got #{name} attr" RESET<< " by default." << std::endl;
                return #{type}(#{formatted_val});
            }

            std::cout << FG_GOLD_RGB "Got #{name} attr." RESET << std::endl;
            return std::get<#{type}>(it->second);
        }
    CPP
    else
    <<~CPP
        #{type} #{name}() const  {
            auto it = node_->attrs_.find("#{name}");
            if (it == node_->attrs_.end()) {
                throw Core::BebraErr("Missing #{name} at #{op_name}!");
            }

            std::cout << FG_GOLD_RGB "Got #{name} attr" << RESET << std::endl;
            return std::get<#{type}>(it->second);
        }
    CPP
    end
end

def generate_op(op)

    name = op["name"]
    attrs = []
    if op["attributes"]
        attrs = op["attributes"].map { |attr| generate_attr(attr, name) }
    end
    <<~CPP
    struct Op#{name} {
        const Core::BebraNode* node_;

        explicit Op#{name}(const Core::BebraNode* node) : node_(node) {
            if (node_->op_type_ != "#{name}") {
                throw Core::BebraErr("Not a #{name} node...");
            }
            std::cout << UNDERLINE_GREEN "Got #{name} node!" RESET << std::endl;
        }
        #{attrs.join("\n")}
        };
    CPP

end

def main

    ops = YAML.load_file(YAML_PATH)
    code_gen = File.open("BebraOperators.hpp", "w")

    code_gen.write(generate_header())

    ops.each do |op|
        code_gen.write(generate_op(op))
    end

    code_gen.write(generate_footer())

    puts "Generated #{ops.size} operations!"

end


main if __FILE__ == $0
