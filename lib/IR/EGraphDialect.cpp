#include "MLIREGraph/IR/EGraphDialect.h"
#include "MLIREGraph/IR/EGraphOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::egraph;

#include "MLIREGraph/IR/EGraphOpsDialect.cpp.inc"

void EGraphDialect::initialize() {
  // Register all egraph IR operations; the dialect does not own extra
  // in-memory state beyond the generated op definitions.
  addOperations<
#define GET_OP_LIST
#include "MLIREGraph/IR/EGraphOps.cpp.inc"
      >();
}

Type EGraphDialect::parseType(DialectAsmParser &parser) const {
  // v1.1 keeps persistent e-class identity in symbols, so the dialect should
  // reject attempts to spell legacy custom handle types in assembly.
  parser.emitError(parser.getCurrentLocation(),
                   "egraph dialect does not define custom types");
  return Type();
}

void EGraphDialect::printType(Type type, DialectAsmPrinter &printer) const {
  (void)type;
  (void)printer;
  llvm_unreachable("egraph dialect does not define custom types");
}
