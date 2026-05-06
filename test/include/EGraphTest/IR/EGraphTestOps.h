#ifndef MLIR_EGRAPH_TEST_IR_EGRAPHTESTOPS_H
#define MLIR_EGRAPH_TEST_IR_EGRAPHTESTOPS_H

/// Declares the test-only operations and includes the generated op classes.
#include "EGraphTest/IR/EGraphTestDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "EGraphTest/IR/EGraphTestOps.h.inc"

#endif // MLIR_EGRAPH_TEST_IR_EGRAPHTESTOPS_H
