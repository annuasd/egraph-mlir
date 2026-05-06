# MLIR-EGraph

MLIR-EGraph is an MLIR-native e-graph implementation inspired by
[egg](https://github.com/egraphs-good/egg). It provides a symbolic e-graph
dialect, rewrite and saturation infrastructure, and extraction backends for
rebuilding optimized MLIR IR.

## Build

```sh
cmake -S . -B build \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --target mlir-egraph-opt mlir-egraph-demo
```

Enable the Z3-backed extractor when a Z3 CMake package is available:

```sh
cmake -S . -B build \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
  -DMLIR_EGRAPH_ENABLE_Z3=ON
```

## Quick Start

Import a `func.func` into an e-graph:

```sh
build/bin/mlir-egraph-opt test/Smoke/func-to-egraph.mlir --convert-func-to-egraph
```

Run the standalone demo:

```sh
build/bin/mlir-egraph-demo
```

## License

MLIR-EGraph is licensed under the MIT License. See [LICENSE](LICENSE).
