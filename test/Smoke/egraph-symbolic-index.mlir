// RUN: mlir-egraph-opt %s --test-egraph-symbolic-index 2>&1 | FileCheck %s

// This smoke test exercises the symbolic index on three small cases:
// parent-use lookup for an input, structural hash-cons reuse for identical
// candidate refs, and structural separation when the child symbol changes.
module {
  egraph.egraph @symbolic_index(%x: i32, %y: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.input @y = %y : i32

    egraph.eclass @c2 : i32 {
      candidate args() {
        %c2 = arith.constant 2 : i32
        egraph.yield %c2 : i32
      }
    }

    egraph.eclass @mul_a : i32 {
      candidate args(@x, @c2) (%lhs: i32, %rhs: i32) {
        %mul = arith.muli %lhs, %rhs : i32
        egraph.yield %mul : i32
      }
    }

    // This candidate intentionally duplicates @mul_a so the test pass can
    // observe structural hash-cons reuse without changing child symbols.
    egraph.eclass @mul_b : i32 {
      candidate args(@x, @c2) (%lhs: i32, %rhs: i32) {
        %mul = arith.muli %lhs, %rhs : i32
        egraph.yield %mul : i32
      }
    }

    // This candidate keeps the same payload op shape but swaps the first child
    // symbol, so it must remain distinct in the structural index.
    egraph.eclass @mul_y : i32 {
      candidate args(@y, @c2) (%lhs: i32, %rhs: i32) {
        %mul = arith.muli %lhs, %rhs : i32
        egraph.yield %mul : i32
      }
    }

    return @mul_a : i32
  }
}

// CHECK: remark: parent index for @x: 2 candidate(s)
// CHECK: remark: structural hashcons reused @mul_a
// CHECK: remark: structural key kept different child symbols separate
// CHECK: remark: rebuilt parent index for @x: 2 candidate(s)
