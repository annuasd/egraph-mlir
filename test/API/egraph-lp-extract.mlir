// REQUIRES: z3
// RUN: mlir-egraph-opt %s --test-egraph-extract-lp 2>&1 | FileCheck %s

module {
  // Exercise Z3-backed global selection, deterministic tie-break, alias forwarding, and cycle rejection.
  egraph.egraph @lp_extract(%x: i32) -> (i32, i32, i32, i32) {
    egraph.input @x = %x : i32

    egraph.eclass @a : i32 {
      candidate args() {
        %c1 = arith.constant 1 : i32
        egraph.yield %c1 : i32
      }
    }

    egraph.eclass @b : i32 {
      candidate args() {
        %c1 = arith.constant 1 : i32
        egraph.yield %c1 : i32
      }
    }

    egraph.eclass @shared : i32 {
      candidate args() {
        %c5 = arith.constant 5 : i32
        egraph.yield %c5 : i32
      }
    }

    egraph.eclass @root1 : i32 {
      candidate args(@a, @b) (%lhs: i32, %rhs: i32) {
        %add = arith.addi %lhs, %rhs : i32
        egraph.yield %add : i32
      }

      candidate args(@shared, @x) (%value: i32, %input: i32) {
        %mul = arith.muli %value, %input : i32
        egraph.yield %mul : i32
      }
    }

    egraph.eclass @root2 : i32 {
      candidate args(@shared, @x) (%value: i32, %input: i32) {
        %add = arith.addi %value, %input : i32
        egraph.yield %add : i32
      }
    }

    egraph.eclass @tie_root : i32 {
      candidate args(@x) (%input: i32) {
        %add = arith.addi %input, %input : i32
        egraph.yield %add : i32
      }

      candidate args(@x) (%input: i32) {
        %mul = arith.muli %input, %input : i32
        egraph.yield %mul : i32
      }
    }

    egraph.eclass @input_alias_leaf : i32 {
      candidate args(@x) (%value: i32) {
        egraph.yield %value : i32
      }
    }

    egraph.eclass @input_alias_root : i32 {
      candidate args(@input_alias_leaf) (%value: i32) {
        egraph.yield %value : i32
      }
    }

    egraph.eclass @cycle : i32 {
      candidate args(@cycle, @x) (%self: i32, %input: i32) {
        %add = arith.addi %self, %input : i32
        egraph.yield %add : i32
      }
    }

    return @root1, @root2, @tie_root, @input_alias_root : i32, i32, i32, i32
  }
}

// CHECK: remark: lp extract selected order -> @shared, @root1, @root2, @tie_root
// CHECK: remark: lp extract alias order -> @input_alias_leaf, @input_alias_root
// CHECK: remark: lp extract selected @shared candidate=arith.constant local_cost=5 subtree_cost=5
// CHECK: remark: lp extract selected @root1 candidate=arith.muli local_cost=2 subtree_cost=7
// CHECK: remark: lp extract selected @root2 candidate=arith.addi local_cost=1 subtree_cost=6
// CHECK: remark: lp extract selected @tie_root candidate=arith.addi local_cost=1 subtree_cost=1
// CHECK: remark: lp extract root costs -> @root1=7, @root2=6, @tie_root=1, @input_alias_root=0 global_cost=9
// CHECK: remark: lp extract rejected cyclic root
