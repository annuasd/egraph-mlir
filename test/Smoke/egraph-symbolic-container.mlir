// RUN: mlir-egraph-opt %s | FileCheck %s

module {
  // Empty and argument-only graphs cover container shape without payload edges.
  egraph.egraph @empty() {
    egraph.return
  }

  egraph.egraph @with_args(%lhs: i32, %rhs: i64) {
    egraph.return
  }

  egraph.egraph @with_input(%x: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.return @x : i32
  }

  // eclass and cycle cases cover child symbol references, alias candidates, and self references.
  egraph.egraph @with_eclass(%lhs: i32, %rhs: i32) -> i32 {
    egraph.input @lhs = %lhs : i32
    egraph.input @rhs = %rhs : i32
    egraph.eclass @sum : i32 {
      candidate args(@lhs, @rhs) (%a: i32, %b: i32) {
        %v = egraph_test.op_a %a, %b : (i32, i32) -> i32
        egraph.yield %v : i32
      }
      candidate args(@lhs) (%a: i32) {
        egraph.yield %a : i32
      }
    }
    egraph.return @sum : i32
  }

  // Mixed self and mutual references keep the printer and verifier honest on cyclic graphs.
  egraph.egraph @with_cycles(%x: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.eclass @self : i32 {
      candidate args(@self) (%a: i32) {
        egraph.yield %a : i32
      }
      candidate args(@x) (%a: i32) {
        egraph.yield %a : i32
      }
    }
    egraph.eclass @lhs : i32 {
      candidate args(@rhs) (%a: i32) {
        egraph.yield %a : i32
      }
    }
    egraph.eclass @rhs : i32 {
      candidate args(@lhs) (%a: i32) {
        egraph.yield %a : i32
      }
    }
    egraph.return @rhs : i32
  }
}

// CHECK-LABEL: module {
// CHECK: egraph.egraph @empty() {
// CHECK-NEXT:   return
// CHECK-NEXT: }
// CHECK: egraph.egraph @with_args(%arg0: i32, %arg1: i64) {
// CHECK-NEXT:   return
// CHECK-NEXT: }
// CHECK: egraph.egraph @with_input(%arg0: i32) -> i32 {
// CHECK-NEXT:   input @x = %arg0 : i32
// CHECK-NEXT:   return @x : i32
// CHECK-NEXT: }
// CHECK: egraph.egraph @with_eclass(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:   input @lhs = %arg0 : i32
// CHECK-NEXT:   input @rhs = %arg1 : i32
// CHECK-NEXT:   eclass @sum : i32 {
// CHECK-NEXT:     candidate args(@lhs, @rhs) (%arg2: i32, %arg3: i32) {
// CHECK-NEXT:       %0 = egraph_test.op_a %arg2, %arg3 : (i32, i32) -> i32
// CHECK-NEXT:       egraph.yield %0 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     candidate args(@lhs) (%arg2: i32) {
// CHECK-NEXT:       egraph.yield %arg2 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return @sum : i32
// CHECK-NEXT: }
// CHECK: egraph.egraph @with_cycles(%arg0: i32) -> i32 {
// CHECK-NEXT:   input @x = %arg0 : i32
// CHECK-NEXT:   eclass @self : i32 {
// CHECK-NEXT:     candidate args(@self) ([[SELF:%arg[0-9]+]]: i32) {
// CHECK-NEXT:       egraph.yield [[SELF]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     candidate args(@x) ([[X:%arg[0-9]+]]: i32) {
// CHECK-NEXT:       egraph.yield [[X]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   eclass @lhs : i32 {
// CHECK-NEXT:     candidate args(@rhs) ([[LHS:%arg[0-9]+]]: i32) {
// CHECK-NEXT:       egraph.yield [[LHS]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   eclass @rhs : i32 {
// CHECK-NEXT:     candidate args(@lhs) ([[RHS:%arg[0-9]+]]: i32) {
// CHECK-NEXT:       egraph.yield [[RHS]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return @rhs : i32
// CHECK-NEXT: }
