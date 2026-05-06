// RUN: mlir-egraph-opt %s --test-egraph-extract-cost-model 2>&1 | FileCheck %s

module {
  // Exercise single-node cost selection across multiple candidates in one eclass.
  egraph.egraph @extract_cost(%x: i32) -> i32 {
    egraph.input @x = %x : i32

    egraph.eclass @choice : i32 {
      candidate args(@x) (%arg: i32) {
        %mul = arith.muli %arg, %arg : i32
        egraph.yield %mul : i32
      }

      candidate args(@x) (%arg: i32) {
        %add = arith.addi %arg, %arg : i32
        egraph.yield %add : i32
      }
    }

    return @choice : i32
  }
}

// CHECK-DAG: remark: extract cost selection -> @choice candidate=arith.muli cost=1 subtree=1 root_cost=1 mode=lp
