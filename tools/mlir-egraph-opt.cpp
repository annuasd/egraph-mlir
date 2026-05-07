#include "MLIREGraph/IR/EGraphDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#ifdef MLIR_EGRAPH_ENABLE_TEST_DIALECT
void registerEGraphTestPasses();
#endif

#ifdef MLIR_EGRAPH_ENABLE_TEST_DIALECT
#include "EGraphTest/IR/EGraphTestDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#endif

#ifndef MLIR_EGRAPH_TEST_LIBRARY
int main(int argc, char **argv) {
#ifdef MLIR_EGRAPH_ENABLE_TEST_DIALECT
  // Keep test-only pass registration behind the dedicated build flag.
  registerEGraphTestPasses();
#endif

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::egraph::EGraphDialect>();
  registry.insert<mlir::func::FuncDialect>();
#ifdef MLIR_EGRAPH_ENABLE_TEST_DIALECT
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::egraph::test::EGraphTestDialect>();
#endif
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR-EGraph optimizer driver\n", registry));
}
#endif
