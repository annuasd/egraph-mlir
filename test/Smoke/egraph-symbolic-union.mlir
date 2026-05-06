// RUN: mlir-egraph-opt %s --test-egraph-symbolic-union 2>&1 | FileCheck %s

module {
  egraph.egraph @symbolic_union(%x: i32) -> i32 {
    egraph.input @x = %x : i32

    // The leader should absorb the rhs candidate before rebuild, then keep
    // the original rhs payload available through the leader's defs.
    egraph.eclass @lhs : i32 {
      candidate args(@x) (%value: i32) {
        %v = egraph_test.op_b %value {tag = "lhs"} : (i32) -> i32
        egraph.yield %v : i32
      }
    }

    egraph.eclass @rhs : i32 {
      candidate args(@x) (%value: i32) {
        %v = egraph_test.op_a %value, %value {mode = "rhs"} : (i32, i32) -> i32
        egraph.yield %v : i32
      }
    }

    egraph.eclass @user : i32 {
      candidate args(@rhs) (%value: i32) {
        %v = egraph_test.op_b %value {tag = "user"} : (i32) -> i32
        egraph.yield %v : i32
      }
    }

    return @rhs : i32
  }
}

// These checks cover the leader rewrite, rebuild, and parent requeue flow.
// CHECK-DAG: remark: symbolic union resolved @rhs to leader @lhs before rebuild
// CHECK-DAG: remark: symbolic union rebuild returned affected parent candidates
// CHECK-DAG: remark: symbolic union leader defs included original rhs candidate
// CHECK-DAG: remark: symbolic union rebuild removed absorbed rhs eclass
// CHECK: egraph.egraph @symbolic_union(%arg0: i32) -> i32 {
// CHECK: input @x = %arg0 : i32
// CHECK-NOT: eclass @rhs : i32 {
// CHECK: eclass @lhs : i32 {
// CHECK: candidate args(@x) (%arg{{[0-9]+}}: i32) {
// CHECK: egraph_test.op_b %arg{{[0-9]+}} {tag = "lhs"} : (i32) -> i32
// CHECK: candidate args(@x) (%[[RHS_ARG:arg[0-9]+]]: i32) {
// CHECK: egraph_test.op_a %[[RHS_ARG]], %[[RHS_ARG]] {mode = "rhs"} : (i32, i32) -> i32
// CHECK-NOT: eclass @rhs : i32 {
// CHECK: eclass @user : i32 {
// CHECK: candidate args(@lhs) (%[[USER_ARG:arg[0-9]+]]: i32) {
// CHECK: egraph_test.op_b %[[USER_ARG]] {tag = "user"} : (i32) -> i32
// CHECK-NOT: eclass @rhs : i32 {
// CHECK: return @lhs : i32
