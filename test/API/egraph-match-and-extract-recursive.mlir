// RUN: mlir-egraph-opt %s --test-egraph-match-and-extract-recursive 2>&1 | FileCheck %s --check-prefix=REMARK
// RUN: mlir-egraph-opt %s --test-egraph-match-and-extract-recursive 2>/dev/null | FileCheck %s --check-prefix=IR

module {
  func.func @nested_example(%x: i32, %ub: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %x) -> (i32) {
      %c2 = arith.constant 2 : i32
      %mul = arith.muli %acc, %c2 : i32
      %div = arith.divsi %mul, %c2 : i32
      scf.yield %div : i32
    }
    return %result : i32
  }
}

// REMARK: remark: egraph recursive pipeline matched and extracted nested arith example

// IR: func.func @nested_example(%[[ARG0:arg[0-9]+]]: i32, %[[UB:arg[0-9]+]]: index) -> i32 {
// IR:   %[[C0:.*]] = arith.constant 0 : index
// IR:   %[[C1:.*]] = arith.constant 1 : index
// IR:   %[[RESULT:.*]] = scf.for %{{.*}} = %[[C0]] to %[[UB]] step %[[C1]] iter_args(%[[ACC:arg[0-9]+]] = %[[ARG0]]) -> (i32) {
// IR:     scf.yield %[[ACC]] : i32
// IR:   }
// IR-NOT: arith.muli
// IR-NOT: arith.divsi
// IR:   return %[[RESULT]] : i32
// IR: }
