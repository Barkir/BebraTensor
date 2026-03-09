YAML_PATH = "ops.yaml"
YAML_TO_CPP_TYPE_HASH = {
    "int64[]" => "std::vector<int64_t>",
    "int64" => "int64_t",
    "float32[]" => "std::vector<float>",
    "float32" => "float",
    "string" => "std::string"
}
