// RUN: mlir-egraph-opt %s --test-egraph-greedy-extract 2>&1 | FileCheck %s

module {
  // Exercise a rebuilt input alias that still points at the original input symbol.
  egraph.egraph @rebuilt_input_alias(%x: i32) -> i32 {
    egraph.input @x = %x : i32

    egraph.eclass @alias : i32 {
      candidate args(@x) (%value: i32) {
        egraph.yield %value : i32
      }
    }

    return @alias : i32
  }
}

// CHECK: remark: rebuilt input alias greedy extract preserved @alias -> @x cost=0
