// RUN: transpose-example %s | FileCheck %s

module {
  func.func @transpose_example(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 1, 0>}
        : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %1 = tosa.sin %0
        : (tensor<3x2xf32>) -> tensor<3x2xf32>
    %2 = tosa.transpose %1 {perms = array<i32: 1, 0>}
        : (tensor<3x2xf32>) -> tensor<2x3xf32>
    return %2 : tensor<2x3xf32>
  }
}

// CHECK-LABEL: func.func @transpose_example(
// CHECK-NOT: tosa.transpose
// CHECK: %[[SIN:.*]] = tosa.sin %arg0 : (tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK: return %[[SIN]] : tensor<2x3xf32>
