# MLIR-EGraph

MLIR-EGraph is an MLIR-native e-graph implementation inspired by
[egg](https://github.com/egraphs-good/egg). It provides a symbolic e-graph
dialect, rewrite and saturation infrastructure, and extraction backends for
rebuilding optimized MLIR IR.

## Build

```sh
cmake -S . -B build 
# Use Z3 for graph extraction.
# -DMLIR_EGRAPH_ENABLE_Z3=ON
# Use OR-Tools for graph extraction.
# -DMLIR_EGRAPH_ENABLE_OR_TOOLS=ON

# Run tests.
cmake --build build --target check-mlir-egraph

# Build examples.
cmake --build build --target mlir-egraph-example
```

## Quick Start

Run the standalone block-level rewrite and extract examples:

```sh
build/bin/arith-example
build/bin/auto-vectorize-example
build/bin/transpose-example
```

## License

MLIR-EGraph is licensed under the MIT License. See [LICENSE](LICENSE).
