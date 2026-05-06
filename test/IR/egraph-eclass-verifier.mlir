// RUN: mlir-egraph-opt %s | FileCheck %s

module {
  egraph.egraph @valid_symbolic_verifier_edges(%x: i32) -> i32 {
    egraph.input @x = %x : i32

    // A standalone constant candidate keeps the zero-operand case visible.
    egraph.eclass @constant : i32 {
      candidate args() {
        %v = arith.constant 1 : i32
        egraph.yield %v : i32
      }
    }

    // Self-reference and input reference are both legal symbol edges.
    egraph.eclass @self : i32 {
      candidate args(@self) (%value: i32) {
        egraph.yield %value : i32
      }
      candidate args(@x) (%value: i32) {
        egraph.yield %value : i32
      }
    }

    // Mutual references are accepted as long as symbol lookup stays valid.
    egraph.eclass @lhs : i32 {
      candidate args(@rhs) (%value: i32) {
        egraph.yield %value : i32
      }
    }

    egraph.eclass @rhs : i32 {
      candidate args(@lhs) (%value: i32) {
        egraph.yield %value : i32
      }
      candidate args(@constant) (%value: i32) {
        egraph.yield %value : i32
      }
    }

    egraph.return @rhs : i32
  }
}

// CHECK-LABEL: egraph.egraph @valid_symbolic_verifier_edges
// CHECK: eclass @constant : i32 {
// CHECK: candidate args() {
// CHECK: arith.constant 1 : i32
// CHECK: eclass @self : i32 {
// CHECK: candidate args(@self) ([[SELF:%arg[0-9]+]]: i32) {
// CHECK: egraph.yield [[SELF]] : i32
// CHECK: candidate args(@x) ([[X:%arg[0-9]+]]: i32) {
// CHECK: egraph.yield [[X]] : i32
// CHECK: eclass @lhs : i32 {
// CHECK: candidate args(@rhs) ([[LHS:%arg[0-9]+]]: i32) {
// CHECK: egraph.yield [[LHS]] : i32
// CHECK: eclass @rhs : i32 {
// CHECK: candidate args(@lhs) ([[RHS:%arg[0-9]+]]: i32) {
// CHECK: egraph.yield [[RHS]] : i32
// CHECK: candidate args(@constant) ([[C:%arg[0-9]+]]: i32) {
// CHECK: egraph.yield [[C]] : i32
// CHECK: return @rhs : i32
