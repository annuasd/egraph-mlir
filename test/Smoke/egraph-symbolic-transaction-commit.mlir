// RUN: mlir-egraph-opt %s --test-egraph-symbolic-transaction-commit 2>&1 | FileCheck %s

// This fixture keeps the transaction-commit path focused on scratch intern,
// symbolic merge, and affected-parent reporting.

module {
  egraph.egraph @commit(%x: i32) -> i32 {
    egraph.input @x = %x : i32

    egraph.eclass @c2 : i32 {
      candidate args() {
        %value = arith.constant 2 : i32
        egraph.yield %value : i32
      }
    }

    egraph.eclass @mul : i32 {
      candidate args(@x, @c2) (%lhs: i32, %rhs: i32) {
        %value = arith.muli %lhs, %rhs : i32
        egraph.yield %value : i32
      }
    }

    egraph.eclass @use : i32 {
      candidate args(@mul, @c2) (%lhs: i32, %rhs: i32) {
        %value = arith.addi %lhs, %rhs : i32
        egraph.yield %value : i32
      }
    }

    egraph.return @use : i32
  }
}

// The CHECK lines below cover the commit event and the parent candidates it
// reports after the symbolic merge.
// CHECK-DAG: remark: symbolic commit interned scratch DAG
// CHECK-DAG: remark: symbolic commit merged shli candidate into @mul
// CHECK-DAG: remark: symbolic commit returned affected parent candidates
