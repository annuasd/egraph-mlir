// RUN: mlir-egraph-opt %s | FileCheck %s
// Smoke test for the tool's default round-trip on an empty module.

// CHECK-LABEL: module
// CHECK-NEXT: }
module {
}
