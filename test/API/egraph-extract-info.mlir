// RUN: mlir-egraph-opt %s --test-egraph-extract-info 2>&1 | FileCheck %s

module {
  // Exercise default root selection, explicit root overrides, and dirty-graph rejection.
  egraph.egraph @extract_info(%x: i32) -> (i32, i32) {
    egraph.input @x = %x : i32

    egraph.eclass @lhs : i32 {
      candidate args(@x) (%arg: i32) {
        egraph.yield %arg : i32
      }
    }

    egraph.eclass @rhs : i32 {
      candidate args() {
        %c1 = arith.constant 1 : i32
        egraph.yield %c1 : i32
      }
    }

    egraph.eclass @user : i32 {
      candidate args(@lhs, @rhs) (%lhs_value: i32, %rhs_value: i32) {
        %sum = arith.addi %lhs_value, %rhs_value : i32
        egraph.yield %sum : i32
      }
    }

    return @rhs, @user : i32, i32
  }
}

// CHECK-DAG: remark: extract info default roots -> @rhs, @user mode=greedy
// CHECK-DAG: remark: extract info explicit roots -> @lhs, @user mode=lp
// CHECK-DAG: remark: extract info rejected dirty graph
