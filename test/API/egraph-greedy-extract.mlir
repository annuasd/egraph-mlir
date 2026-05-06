// RUN: mlir-egraph-opt %s --test-egraph-greedy-extract 2>&1 | FileCheck %s

module {
  // Exercise recursive greedy selection, alias forwarding, and cycle rejection.
  egraph.egraph @greedy_extract(%x: i32, %y: i32) -> (i32, i32, i32) {
    egraph.input @x = %x : i32
    egraph.input @y = %y : i32

    egraph.eclass @cheap : i32 {
      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        %add = arith.addi %lhs, %rhs : i32
        egraph.yield %add : i32
      }
    }

    egraph.eclass @expensive : i32 {
      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        %sub = arith.subi %lhs, %rhs : i32
        egraph.yield %sub : i32
      }
    }

    egraph.eclass @root : i32 {
      candidate args(@expensive, @x) (%lhs: i32, %rhs: i32) {
        %add = arith.addi %lhs, %rhs : i32
        egraph.yield %add : i32
      }

      candidate args(@cheap, @x) (%lhs: i32, %rhs: i32) {
        %mul = arith.muli %lhs, %rhs : i32
        egraph.yield %mul : i32
      }
    }

    egraph.eclass @loop : i32 {
      candidate args(@loop, @x) (%self: i32, %rhs: i32) {
        %add = arith.addi %self, %rhs : i32
        egraph.yield %add : i32
      }

      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        %mul = arith.muli %lhs, %rhs : i32
        egraph.yield %mul : i32
      }
    }

    egraph.eclass @alias_leaf : i32 {
      candidate args(@cheap) (%value: i32) {
        egraph.yield %value : i32
      }
    }

    egraph.eclass @alias_root : i32 {
      candidate args(@alias_leaf) (%value: i32) {
        egraph.yield %value : i32
      }
    }

    egraph.eclass @only_cycle : i32 {
      candidate args(@only_cycle, @x) (%self: i32, %rhs: i32) {
        %add = arith.addi %self, %rhs : i32
        egraph.yield %add : i32
      }
    }

    return @root, @loop, @alias_root : i32, i32, i32
  }
}

// CHECK: remark: greedy extract selected order -> @cheap, @root, @loop
// CHECK: remark: greedy extract alias order -> @alias_leaf, @alias_root
// CHECK: remark: greedy extract selected @cheap candidate=arith.addi local_cost=1 subtree_cost=1
// CHECK: remark: greedy extract selected @root candidate=arith.muli local_cost=4 subtree_cost=5
// CHECK: remark: greedy extract selected @loop candidate=arith.muli local_cost=4 subtree_cost=4
// CHECK: remark: greedy extract rejected cyclic-only root
