#include "EGraphTest/IR/EGraphTestDialect.h"
#include "EGraphTest/IR/EGraphTestOps.h"

using namespace mlir;
using namespace mlir::egraph::test;

#include "EGraphTest/IR/EGraphTestOpsDialect.cpp.inc"

void EGraphTestDialect::initialize() {
  // Register the generated test ops in one place to match MLIR dialect style.
  addOperations<
#define GET_OP_LIST
#include "EGraphTest/IR/EGraphTestOps.cpp.inc"
      >();
}
