# MLIR-EGraph

MLIR-EGraph is an MLIR-native e-graph implementation inspired by
[egg](https://github.com/egraphs-good/egg). It provides a symbolic e-graph
dialect, rewrite and saturation infrastructure, and extraction backends for
rebuilding optimized MLIR IR.

## Build

```sh
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
# Use Z3 for graph extraction.
# -DMLIR_EGRAPH_ENABLE_Z3=ON
# If LLVM/MLIR/Z3 are not in the default CMake search path, add:
# -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir -DZ3_DIR=/path/to/z3/lib/cmake/z3

# Run tests.
cmake --build build --target check-mlir-egraph

# Build examples.
cmake --build build --target mlir-egraph-example
```

## Quick Start

Run the standalone block-level rewrite and extract examples:

```sh
build/bin/arith-example
build/bin/transpose-example
```

## License

MLIR-EGraph is licensed under the MIT License. See [LICENSE](LICENSE).
