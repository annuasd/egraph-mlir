// RUN: mlir-egraph-opt %s --test-egraph-match-and-extract 2>&1 | FileCheck %s --check-prefix=REMARK
// RUN: mlir-egraph-opt %s --test-egraph-match-and-extract 2>/dev/null | FileCheck %s --check-prefix=IR

module {
  // Exercise the combined match and extract entry on a simple arithmetic demo.
  func.func @arith_demo(%x: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %mul = arith.muli %x, %c2 : i32
    %div = arith.divsi %mul, %c2 : i32
    return %div : i32
  }
}

// REMARK: remark: egraph pipeline matched and extracted arith demo

// IR: func.func @arith_demo(%[[ARG0:arg[0-9]+]]: i32) -> i32 {
// IR: return %[[ARG0]] : i32
// IR-NOT: arith.constant
// IR-NOT: arith.muli
// IR-NOT: arith.divsi
// IR: }
