// RUN: mlir-egraph-opt %s --convert-func-to-egraph | FileCheck %s

// Keep this coverage focused on basic func-to-egraph import shapes.
module {
  // This function covers tuple return forwarding into symbolic egraph returns.
  func.func @forward(%x: i32, %y: i64) -> (i64, i32) {
    return %y, %x : i64, i32
  }

  // This function covers importing a small arithmetic DAG into symbolic eclasses.
  func.func @arith_demo(%x: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %mul = arith.muli %x, %c2 : i32
    %div = arith.divsi %mul, %c2 : i32
    return %div : i32
  }

  // Keep a pre-existing egraph to exercise generated symbol collision handling.
  egraph.egraph @naming_conflict_egraph(%arg0: i32) -> i32 {
    input @arg0 = %arg0 : i32
    return @arg0 : i32
  }

  func.func @naming_conflict(%x: i32) -> i32 {
    return %x : i32
  }
}

// The first check group keeps tuple result ordering intact across the imported egraph.
// CHECK-LABEL: module {
// CHECK: func.func @forward(%arg0: i32, %arg1: i64) -> (i64, i32) {
// CHECK-NEXT:   return %arg1, %arg0 : i64, i32
// CHECK-NEXT: }
// CHECK-NEXT: egraph.egraph @forward_egraph(%arg0: i32, %arg1: i64) -> (i64, i32) {
// CHECK-NEXT:   input @arg0 = %arg0 : i32
// CHECK-NEXT:   input @arg1 = %arg1 : i64
// CHECK-NEXT:   return @arg1, @arg0 : i64, i32
// CHECK-NEXT: }

// The second check group keeps the imported arithmetic chain split across symbolic eclasses.
// CHECK-NEXT: func.func @arith_demo(%arg0: i32) -> i32 {
// CHECK-NEXT:   %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:   %0 = arith.muli %arg0, %c2_i32 : i32
// CHECK-NEXT:   %1 = arith.divsi %0, %c2_i32 : i32
// CHECK-NEXT:   return %1 : i32
// CHECK-NEXT: }
// CHECK-NEXT: egraph.egraph @arith_demo_egraph(%arg0: i32) -> i32 {
// CHECK-NEXT:   input @arg0 = %arg0 : i32
// CHECK-NEXT:   eclass @c2_i32 : i32 {
// CHECK-NEXT:     candidate args() {
// CHECK-NEXT:       %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:       egraph.yield %c2_i32 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   eclass @muli : i32 {
// CHECK-NEXT:     candidate args(@arg0, @c2_i32) ([[MUL_LHS:%arg[0-9]+]]: i32, [[MUL_RHS:%arg[0-9]+]]: i32) {
// CHECK-NEXT:       [[MUL:%[0-9]+]] = arith.muli [[MUL_LHS]], [[MUL_RHS]] : i32
// CHECK-NEXT:       egraph.yield [[MUL]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   eclass @divsi : i32 {
// CHECK-NEXT:     candidate args(@muli, @c2_i32) ([[DIV_LHS:%arg[0-9]+]]: i32, [[DIV_RHS:%arg[0-9]+]]: i32) {
// CHECK-NEXT:       [[DIV:%[0-9]+]] = arith.divsi [[DIV_LHS]], [[DIV_RHS]] : i32
// CHECK-NEXT:       egraph.yield [[DIV]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return @divsi : i32
// CHECK-NEXT: }

// The last check group keeps generated egraph symbols disambiguated from existing ones.
// CHECK-NEXT: egraph.egraph @naming_conflict_egraph(%arg0: i32) -> i32 {
// CHECK-NEXT:   input @arg0 = %arg0 : i32
// CHECK-NEXT:   return @arg0 : i32
// CHECK-NEXT: }
// CHECK-NEXT: func.func @naming_conflict(%arg0: i32) -> i32 {
// CHECK-NEXT:   return %arg0 : i32
// CHECK-NEXT: }
// CHECK-NEXT: egraph.egraph @naming_conflict_egraph_0(%arg0: i32) -> i32 {
// CHECK-NEXT:   input @arg0 = %arg0 : i32
// CHECK-NEXT:   return @arg0 : i32
// CHECK-NEXT: }
