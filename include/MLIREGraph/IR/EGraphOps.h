#ifndef MLIR_EGRAPH_IR_EGRAPHOPS_H
#define MLIR_EGRAPH_IR_EGRAPHOPS_H

/// Declares the egraph IR operations and includes the generated op classes.
#include "MLIREGraph/IR/EGraphDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "MLIREGraph/IR/EGraphOps.h.inc"

#endif // MLIR_EGRAPH_IR_EGRAPHOPS_H
