// RUN: mlir-egraph-opt %s | FileCheck %s

module {
  egraph.egraph @roundtrip_inputs(%x: i32, %y: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.input @y = %y : i32

    egraph.eclass @sum : i32 {
      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        %v = egraph_test.op_a %lhs, %rhs {mode = "fast"} : (i32, i32) -> i32
        egraph.yield %v : i32
      }
    }

    egraph.return @sum : i32
  }

  egraph.egraph @roundtrip_alias(%x: i32) -> i32 {
    egraph.input @x = %x : i32

    // Alias candidates still yield payload values, not egraph handles.
    egraph.eclass @alias : i32 {
      candidate args(@x) (%value: i32) {
        egraph.yield %value : i32
      }
    }

    egraph.return @alias : i32
  }
}

// CHECK-LABEL: module {
// CHECK-LABEL: egraph.egraph @roundtrip_inputs(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:   input @x = %arg0 : i32
// CHECK-NEXT:   input @y = %arg1 : i32
// CHECK-NEXT:   eclass @sum : i32 {
// CHECK-NEXT:     candidate args(@x, @y) (%arg2: i32, %arg3: i32) {
// CHECK-NEXT:       %[[SUM:.*]] = egraph_test.op_a %arg2, %arg3 {mode = "fast"} : (i32, i32) -> i32
// CHECK-NEXT:       egraph.yield %[[SUM]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return @sum : i32
// CHECK-NEXT: }
// CHECK-LABEL: egraph.egraph @roundtrip_alias(%arg0: i32) -> i32 {
// CHECK-NEXT:   input @x = %arg0 : i32
// CHECK-NEXT:   eclass @alias : i32 {
// CHECK-NEXT:     candidate args(@x) (%[[VALUE:arg[0-9]+]]: i32) {
// CHECK-NEXT:       egraph.yield %[[VALUE]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return @alias : i32
// CHECK-NEXT: }
