# BEBRA TENSOR COMPILER
![LLVM Badge](https://img.shields.io/badge/LLVM-262D3A?logo=llvm&logoColor=fff&style=for-the-badge)![C++ Badge](https://img.shields.io/badge/C%2B%2B-00599C?logo=cplusplus&logoColor=fff&style=for-the-badge)![ONNX Badge](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=fff&style=for-the-badge)![Ruby Badge](https://img.shields.io/badge/Ruby-CC342D?logo=ruby&logoColor=fff&style=for-the-badge)
### Installation


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
Project consists of `Core`, `Ops`, `MLIR` parts.

### CORE

The core part consists of `BebraGraph`, `BebraNode`, `BebraTensor`, etc. Model graph consists of tensors and nodes, which are somehow connected through inputs, outputs.

> Each node has op_type. e.g : `Conv`, `Gemm`, `Add`. Some of them have attributes (simpler: constant values)

The specification of operators can be seen [here](https://onnx.ai/onnx/operators/index.html).

### OPS

Op part code is generated using `ruby` and `ops.yaml`.
This approach is good because it is easier to add and delete new instructions.

To generate code -> go to `include/bebra/ops/ruby_gen`.

run this

```
ruby ops_gen.rb
```

Now I have only 7 instructions from official onnx specification, described as .yaml [here](./include/bebra/ops/ruby_gen/ops.yaml)

### GRAPHVIZ
![img](./dot/png/mnist-8.png)

This is a graph for MNIST-8 Neural Network.
To generate graphs for your Neural Networks use `--dump <filename>` syntax when running a program. Then use `python3 dot2png.py` to generate `.png`

##### Usecase for generating dump
```bash
./cmake/bebra_tensor --dump third_party/mnist-8.onnx

python3 dot2png.py
? Choose a .dot file you want to generate .png from (Use arrow keys)
 » dot/mnist-8.dot

```

### MLIR-BACKEND

using mlir api to generate mlir code
you can have a look at it [here](./src/mlir/MLIRPrinter.cpp)

to use it enter `--to-mlir <your_file.onnx>` to compile the model

At the start of compiling we load some dialects: `tosa, arith, func, linalg`

then module is created, then `main` function with input arguments and return values is created.

after generating the code in module we go through some passes in `compileToLLVM`. It starts with `infer shapes` pass which computes dynamic tensor shapes.


-----









