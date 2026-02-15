# BEBRA TENSOR COMPILER

### Installation

maybe you'll need this

```bash
sudo apt install protobuf-compiler libprotobuf-dev
```

```bash
chmod +x ./init.sh
./init.sh
```

#### Running tests
- Tests are written in `tests` directory.
- Run them using `ctest --output-on-failure` in `cmake` dir.

#### Project structure
Project consists of `Core` and `Ops` parts.

### CORE

The core part consists of `BebraGraph`, `BebraNode`, `BebraTensor`, etc. Model graph consists of tensors and nodes, which are somehow connected through inputs, outputs.

> Each node has op_type. e.g : `Conv`, `Gemm`, `Add`. Some of them have attributes (simpler: constant values)

The specification of operators can be seen [here](https://onnx.ai/onnx/operators/index.html).

### OPS

Op part code is generated using `ruby` and `ops.yaml`.
This approach is good because it is easier to add and delete new instructions.

To generate code -> go to `include/bebra/ops`.

run this

```
ruby ops_gen.rb
```

you can also add your instructions, now I have only 7 of them.

-----

#### Future plans

| Task | Stage |
|------|-------|
| Implement static polimorphism. | maybe
| Add graphviz dump | thinking about architecture








