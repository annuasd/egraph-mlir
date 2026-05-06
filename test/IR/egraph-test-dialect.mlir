// RUN: mlir-egraph-opt %s | FileCheck %s

// CHECK-LABEL: func.func @test_ops
func.func @test_ops() {
  // Leaf ops keep stable names so downstream test ops can refer to them.
  // CHECK: %[[X:.*]] = egraph_test.leaf {name = "x"} : i32
  %x = egraph_test.leaf {name = "x"} : i32
  // CHECK: %[[Y:.*]] = egraph_test.leaf {name = "y"} : i32
  %y = egraph_test.leaf {name = "y"} : i32

  // OpA and OpB cover single-result assembly with attrs on producers and users.
  // CHECK: egraph_test.op_a %[[X]], %[[Y]] {mode = "fast"} : (i32, i32) -> i32
  %fast = egraph_test.op_a %x, %y {mode = "fast"} : (i32, i32) -> i32
  // CHECK: egraph_test.op_a %[[X]], %[[Y]] {mode = "accurate"} : (i32, i32) -> i32
  %accurate = egraph_test.op_a %x, %y {mode = "accurate"} : (i32, i32) -> i32
  // CHECK: egraph_test.op_b %{{.*}} {tag = "consumer"} : (i32) -> i32
  %b = egraph_test.op_b %fast {tag = "consumer"} : (i32) -> i32

  // Split and slice ops keep multi-result and derived-result printers visible.
  // CHECK: %[[INPUT:.*]] = egraph_test.leaf {name = "input"} : tensor<4xf32>
  %input = egraph_test.leaf {name = "input"} : tensor<4xf32>
  // CHECK: egraph_test.split %[[INPUT]] : (tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  %s0, %s1 = egraph_test.split %input : (tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  // CHECK: egraph_test.slice0 %[[INPUT]] : (tensor<4xf32>) -> tensor<2xf32>
  %d0 = egraph_test.slice0 %input : (tensor<4xf32>) -> tensor<2xf32>
  // CHECK: egraph_test.slice1 %[[INPUT]] : (tensor<4xf32>) -> tensor<2xf32>
  %d1 = egraph_test.slice1 %input : (tensor<4xf32>) -> tensor<2xf32>

  // The impure op stays in the test dialect so side-effect traits remain covered.
  // CHECK: egraph_test.impure %{{.*}} : (i32) -> i32
  %impure = egraph_test.impure %accurate : (i32) -> i32
  func.return
}
