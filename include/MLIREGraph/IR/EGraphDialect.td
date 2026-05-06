#ifndef MLIR_EGRAPH_DIALECT
#define MLIR_EGRAPH_DIALECT

include "mlir/IR/OpBase.td"

def EGraph_Dialect : Dialect {
  let name = "egraph";
  let summary = "Persistent equality graph dialect";
  let description = [{
    The egraph dialect stores persistent equality graph state in MLIR IR. The
    persistent graph is organized around egraph.egraph containers, named input
    e-classes, candidate-bearing egraph.eclass operations, and egraph.return
    terminators. E-class identity is carried by symbols, while candidate bodies
    contain the payload operations.
  }];
  let cppNamespace = "::mlir::egraph";
  let useDefaultTypePrinterParser = 1;
}

class EGraph_Op<string mnemonic, list<Trait> traits = []>
    : Op<EGraph_Dialect, mnemonic, traits>;

#endif // MLIR_EGRAPH_DIALECT
