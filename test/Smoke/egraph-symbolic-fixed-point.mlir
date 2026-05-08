// RUN: mlir-egraph-opt %s --test-egraph-symbolic-fixed-point 2>&1 | FileCheck %s

module {
  egraph.egraph @symbolic_fixed_point(%x: i32) -> i32 {
    egraph.input @x = %x : i32

    // Two child candidates feed the same input and differ only by tags.
    egraph.eclass @lhs : i32 {
      candidate args(@x) (%value: i32) {
        %v = egraph_test.op_b %value {tag = "lhs-child"} : (i32) -> i32
        egraph.yield %v : i32
      }
    }

    egraph.eclass @rhs : i32 {
      candidate args(@x) (%value: i32) {
        %v = egraph_test.op_b %value {tag = "rhs-child"} : (i32) -> i32
        egraph.yield %v : i32
      }
    }

    // Rebuild should collapse both parent candidates onto the leader child.
    egraph.eclass @parent_lhs : i32 {
      candidate args(@lhs) (%value: i32) {
        %v = egraph_test.op_b %value : (i32) -> i32
        egraph.yield %v : i32
      }
    }

    egraph.eclass @parent_rhs : i32 {
      candidate args(@rhs) (%value: i32) {
        %v = egraph_test.op_b %value : (i32) -> i32
        egraph.yield %v : i32
      }
    }

    // The grandparent should observe the rewritten parent leader as well.
    egraph.eclass @grand_lhs : i32 {
      candidate args(@parent_lhs) (%value: i32) {
        %v = egraph_test.op_b %value : (i32) -> i32
        egraph.yield %v : i32
      }
    }

    egraph.eclass @grand_rhs : i32 {
      candidate args(@parent_rhs) (%value: i32) {
        %v = egraph_test.op_b %value : (i32) -> i32
        egraph.yield %v : i32
      }
    }

    // This probe checks that structural hashing uses rebuilt leader symbols.
    egraph.eclass @probe : i32 {
      candidate args(@rhs, @x) (%lhs: i32, %rhs: i32) {
        %v = egraph_test.op_a %lhs, %rhs {mode = "probe"} : (i32, i32) -> i32
        egraph.yield %v : i32
      }
    }

    // Cycles must survive rebuild without being flattened into stale refs.
    egraph.eclass @self_cycle : i32 {
      candidate args(@self_cycle) (%value: i32) {
        %v = egraph_test.op_b %value {tag = "self"} : (i32) -> i32
        egraph.yield %v : i32
      }
    }

    egraph.eclass @mutual_a : i32 {
      candidate args(@mutual_b) (%value: i32) {
        %v = egraph_test.op_b %value {tag = "mutual"} : (i32) -> i32
        egraph.yield %v : i32
      }
    }

    egraph.eclass @mutual_b : i32 {
      candidate args(@mutual_a) (%value: i32) {
        %v = egraph_test.op_b %value {tag = "mutual"} : (i32) -> i32
        egraph.yield %v : i32
      }
    }

    return @grand_rhs : i32
  }
}

// CHECK-DAG: remark: symbolic rebuild merged parent eclasses after child union
// CHECK-DAG: remark: symbolic rebuild reached grandparent fixed point
// CHECK-DAG: remark: symbolic structural key hashed rebuilt probe by leader symbols
// CHECK-DAG: remark: symbolic rebuild deduplicated transitive member candidates
// CHECK-DAG: remark: symbolic self and mutual cycles survived rebuild
// CHECK: egraph.egraph @symbolic_fixed_point(%arg0: i32) -> i32 {
// CHECK: input @x = %arg0 : i32
// CHECK-NOT: eclass @rhs : i32 {
// CHECK: eclass @parent_lhs : i32 {
// CHECK-NOT: candidate args(@lhs) (%[[PARENT_RHS_ARG:arg[0-9]+]]: i32) {
// CHECK: candidate args(@lhs) (%[[PARENT_LHS_ARG:arg[0-9]+]]: i32) {
// CHECK: egraph_test.op_b %[[PARENT_LHS_ARG]] : (i32) -> i32
// CHECK-NOT: eclass @parent_rhs : i32 {
// CHECK: eclass @grand_lhs : i32 {
// CHECK-NOT: candidate args(@parent_lhs) (%[[GRAND_RHS_ARG:arg[0-9]+]]: i32) {
// CHECK: candidate args(@parent_lhs) (%[[GRAND_LHS_ARG:arg[0-9]+]]: i32) {
// CHECK: egraph_test.op_b %[[GRAND_LHS_ARG]] : (i32) -> i32
// CHECK-NOT: eclass @grand_rhs : i32 {
// CHECK: eclass @probe : i32 {
// CHECK: candidate args(@lhs, @x) (%arg{{[0-9]+}}: i32, %arg{{[0-9]+}}: i32) {
// CHECK: egraph_test.op_a %arg{{[0-9]+}}, %arg{{[0-9]+}} {mode = "probe"} : (i32, i32) -> i32
// CHECK: eclass @self_cycle : i32 {
// CHECK: candidate args(@self_cycle) (%arg{{[0-9]+}}: i32) {
// CHECK: eclass @mutual_a : i32 {
// CHECK: candidate args(@mutual_b) (%arg{{[0-9]+}}: i32) {
// CHECK: eclass @mutual_b : i32 {
// CHECK: candidate args(@mutual_a) (%arg{{[0-9]+}}: i32) {
// CHECK: return @grand_lhs : i32
