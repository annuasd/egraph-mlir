// RUN: mlir-egraph-opt %s | FileCheck %s

module {
  egraph.egraph @assembly_attrs(%x: i32, %y: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.input @y = %y : i32

    egraph.eclass @sum : i32 attributes {tag = "primary"} {
      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        %v = egraph_test.op_a %lhs, %rhs {mode = "sum"} : (i32, i32) -> i32
        egraph.yield %v : i32
      }
      candidate args(@x) (%alias: i32) {
        egraph.yield %alias : i32
      }
    }

    egraph.eclass @consumer : i32 {
      candidate args(@sum) (%input: i32) {
        %v = egraph_test.op_b %input {tag = "consumer"} : (i32) -> i32
        egraph.yield %v : i32
      }
    }

    egraph.return @consumer : i32
  }
}

// CHECK-LABEL: egraph.egraph @assembly_attrs(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:     input @x = %arg0 : i32
// CHECK-NEXT:     input @y = %arg1 : i32
// CHECK-NEXT:     eclass @sum : i32 attributes {tag = "primary"} {
// CHECK-NEXT:       candidate args(@x, @y) (%arg2: i32, %arg3: i32) {
// CHECK-NEXT:         %[[SUM:.*]] = egraph_test.op_a %arg2, %arg3 {mode = "sum"} : (i32, i32) -> i32
// CHECK-NEXT:         egraph.yield %[[SUM]] : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       candidate args(@x) (%arg2: i32) {
// CHECK-NEXT:         egraph.yield %arg2 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     eclass @consumer : i32 {
// CHECK-NEXT:       candidate args(@sum) (%arg2: i32) {
// CHECK-NEXT:         %[[CONSUMER:.*]] = egraph_test.op_b %arg2 {tag = "consumer"} : (i32) -> i32
// CHECK-NEXT:         egraph.yield %[[CONSUMER]] : i32
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return @consumer : i32
// CHECK-NEXT:   }
