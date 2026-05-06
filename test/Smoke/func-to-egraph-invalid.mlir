// RUN: mlir-egraph-opt %s --allow-unregistered-dialect --convert-func-to-egraph -split-input-file -verify-diagnostics
// Keep each negative case focused on one importer rejection.

module {
  // Multi-result pure ops are not imported yet.
  func.func @reject_multi_result(%x: i32) -> i32 {
    %first, %second = egraph_test.split %x : (i32) -> (i32, i32) // expected-error {{'egraph_test.split' op func-to-egraph import currently supports only single-result pure operations}}
    return %first : i32
  }
}

// -----

module {
  // Impure ops must fail import.
  func.func @reject_impure(%x: i32) -> i32 {
    %impure = egraph_test.impure %x : (i32) -> i32 // expected-error {{'egraph_test.impure' op operation is not pure and cannot be imported into egraph}}
    return %impure : i32
  }
}

// -----

module {
  // Control-flow ops are excluded from the importer.
  func.func @has_control_flow(%x: i32) -> i32 {
    cf.br ^bb1(%x : i32) // expected-error {{'cf.br' op control-flow operation cannot be imported into egraph}}
  ^bb1(%arg0: i32):
    return %arg0 : i32
  }
}

// -----

module {
  // Nested regions are also rejected.
  func.func @has_region(%cond: i1, %x: i32, %y: i32) -> i32 {
    "test.region"() ({ // expected-error {{'test.region' op operation with nested regions cannot be imported into egraph}}
    }) : () -> ()
    return %x : i32
  }
}
