require 'yaml'
require_relative "verifiers.rb"
require_relative "constants.rb"
require_relative "helpers.rb"
require_relative "generate.rb"


def write_code_gen(code_gen, ops)
    code_gen.write(generate_header())
    ops.each do |op|
        code_gen.write(generate_op(op))
    end

    code_gen.write(generate_footer())

    puts "Generated #{ops.size} operations!"
end

def write_code_factory(hpp, cpp, ops)
    hpp.write(generate_factory_hpp_header())

        ops.each do |op|
            hpp.write(generate_op_creator_hpp(op))
        end

    hpp.write(generate_factory_hpp_footer())

    cpp.write(generate_factory_cpp_header(ops))
        ops.each do |op|
            cpp.write(generate_op_creator_cpp(op))
        end
    cpp.write(generate_factory_cpp_footer())

end

def write_verify(verifiers, ops)
    verifiers.write(generate_verify_header())
    ops.each do |op|
        verifiers.write(generate_shape_verify(op["category"], op["name"]))
    end
    verifiers.write(generate_verify_footer())
end

def write_variant(op_variant, ops)
    op_variant.write(generate_variant(ops))
end

def main

    ops = YAML.load_file(YAML_PATH)
    code_gen = File.open("../BebraOperators.hpp", "w")
    verifiers = File.open("../../../../src/ops/BebraOperators.cpp", "w")
    code_factory_hpp = File.open("../BebraFactory.hpp", "w")
    code_factory_cpp = File.open("../../../../src/ops/BebraFactory.cpp", "w")
    op_variant = File.open("../BebraVariant.hpp", "w")

    write_code_gen(code_gen, ops)
    write_code_factory(code_factory_hpp, code_factory_cpp, ops)
    write_variant(op_variant, ops)
    write_verify(verifiers, ops)


end


main if __FILE__ == $0
