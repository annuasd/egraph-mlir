// RUN: mlir-egraph-opt %s --test-egraph-extract-materialization 2>&1 | FileCheck %s --check-prefix=REMARK
// RUN: mlir-egraph-opt %s --test-egraph-extract-materialization 2>/dev/null | FileCheck %s --check-prefix=IR

module {
  // Exercise materialization of shared selected eclasses and alias roots into ordinary MLIR.
  egraph.egraph @materialize_demo_egraph(%x: i32, %y: i32) -> (i32, i32) {
    egraph.input @x = %x : i32
    egraph.input @y = %y : i32

    egraph.eclass @shared : i32 {
      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        %add = arith.addi %lhs, %rhs : i32
        egraph.yield %add : i32
      }
    }

    egraph.eclass @root1 : i32 {
      candidate args(@shared, @x) (%value: i32, %input: i32) {
        %mul = arith.muli %value, %input : i32
        egraph.yield %mul : i32
      }
    }

    egraph.eclass @root2 : i32 {
      candidate args(@shared, @y) (%value: i32, %input: i32) {
        %sub = arith.subi %value, %input : i32
        egraph.yield %sub : i32
      }
    }

    egraph.eclass @alias_root : i32 {
      candidate args(@root2) (%value: i32) {
        egraph.yield %value : i32
      }
    }

    return @root1, @alias_root : i32, i32
  }
}

// REMARK: remark: egraph extract materialized @materialize_demo_egraph into @materialize_demo_materialized

// IR: func.func @materialize_demo_materialized(%[[ARG0:arg[0-9]+]]: i32, %[[ARG1:arg[0-9]+]]: i32) -> (i32, i32) {
// IR: %[[SHARED:[0-9]+]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
// IR-NOT: arith.addi
// IR: %[[ROOT1:[0-9]+]] = arith.muli %[[SHARED]], %[[ARG0]] : i32
// IR-NOT: arith.addi
// IR: %[[ROOT2:[0-9]+]] = arith.subi %[[SHARED]], %[[ARG1]] : i32
// IR-NOT: arith.addi
// IR: return %[[ROOT1]], %[[ROOT2]] : i32, i32
// IR: }
