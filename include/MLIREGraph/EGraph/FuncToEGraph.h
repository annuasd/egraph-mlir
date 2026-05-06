#ifndef MLIR_EGRAPH_EGRAPH_FUNCTOEGRAPH_H
#define MLIR_EGRAPH_EGRAPH_FUNCTOEGRAPH_H

#include <memory>

namespace mlir {
class Pass;

namespace egraph {

/// Creates the pass that imports `func.func` into `egraph.egraph`.
std::unique_ptr<Pass> createConvertFuncToEGraphPass();

} // namespace egraph
} // namespace mlir

#endif // MLIR_EGRAPH_EGRAPH_FUNCTOEGRAPH_H
