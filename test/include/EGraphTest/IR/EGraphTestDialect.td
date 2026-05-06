#ifndef MLIR_EGRAPH_TEST_EGRAPHTEST_DIALECT
#define MLIR_EGRAPH_TEST_EGRAPHTEST_DIALECT

include "mlir/IR/OpBase.td"

def EGraphTest_Dialect : Dialect {
  let name = "egraph_test";
  let summary = "Test-only dialect for MLIR-EGraph regression coverage";
  let description = [{
    This dialect provides a small set of test-only operations used by
    MLIR-EGraph regression tests. It intentionally stays under the test tree
    and is not part of the public MLIR-EGraph API.
  }];
  let cppNamespace = "::mlir::egraph::test";
}

class EGraphTest_Op<string mnemonic, list<Trait> traits = []>
    : Op<EGraphTest_Dialect, mnemonic, traits>;

#endif // MLIR_EGRAPH_TEST_EGRAPHTEST_DIALECT
