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
cmake --build build --target mlir-egraph-opt arith-demo transpose-demo
```

Enable the Z3-backed extractor when a Z3 CMake package is available:

```sh
cmake -S . -B build \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
  -DMLIR_EGRAPH_ENABLE_Z3=ON
```

## Quick Start

Run the standalone block-level rewrite and extract demos:

```sh
build/bin/arith-demo
build/bin/transpose-demo
```

## License

MLIR-EGraph is licensed under the MIT License. See [LICENSE](LICENSE).
