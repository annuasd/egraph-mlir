// RUN: mlir-egraph-opt %s --test-egraph-pipeline 2>&1 | FileCheck %s --check-prefix=REMARK
// RUN: mlir-egraph-opt %s --test-egraph-pipeline 2>/dev/null | FileCheck %s --check-prefix=IR

// Keep import remarks and final IR checks in one compact pipeline smoke test.
module {
  // The imported func stays visible beside the derived symbolic egraph.
  func.func @arith_demo(%x: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %mul = arith.muli %x, %c2 : i32
    %div = arith.divsi %mul, %c2 : i32
    return %div : i32
  }
}

// Driver remarks document the pipeline stages exercised by this test.
// REMARK: remark: egraph pipeline imported arith demo
// REMARK: remark: egraph pipeline ran symbolic rewrite driver

// The IR check focuses on the imported function and symbolic egraph output.
// IR-LABEL: module {
// IR: func.func @arith_demo(%[[ARG0:arg[0-9]+]]: i32) -> i32 {
// IR: %[[C2:[0-9A-Za-z_]+]] = arith.constant 2 : i32
// IR: %[[MUL:[0-9]+]] = arith.muli %[[ARG0]], %[[C2]] : i32
// IR: %[[DIV:[0-9]+]] = arith.divsi %[[MUL]], %[[C2]] : i32
// IR: egraph.egraph @arith_demo_egraph(%[[EG_ARG0:arg[0-9]+]]: i32) -> i32 {
// IR-NOT: evalue<
// IR: input @arg0 = %[[EG_ARG0]] : i32
// IR: eclass @c2_i32 : i32 {
// IR: arith.constant 2 : i32
// IR: eclass @muli : i32 {
// IR: candidate args(@divsi, @c2_i32) (%[[MUL_ARG0:arg[0-9]+]]: i32, %[[MUL_ARG1:arg[0-9]+]]: i32) {
// IR: arith.muli %[[MUL_ARG0]], %[[MUL_ARG1]] : i32
// IR: candidate args(@divsi, @__intern_divsi) (%[[SHL_ARG0:arg[0-9]+]]: i32, %[[SHL_ARG1:arg[0-9]+]]: i32) {
// IR: arith.shli %[[SHL_ARG0]], %[[SHL_ARG1]] : i32
// IR-NOT: eclass @__intern_constant : i32 {
// IR-NOT: eclass @__intern_shli : i32 {
// IR: eclass @divsi : i32 {
// IR: candidate args(@muli, @c2_i32) (%[[DIV_ARG0:arg[0-9]+]]: i32, %[[DIV_ARG1:arg[0-9]+]]: i32) {
// IR: arith.divsi %[[DIV_ARG0]], %[[DIV_ARG1]] : i32
// IR: candidate args(@divsi, @__intern_divsi) (%[[DIV_ALIAS0:arg[0-9]+]]: i32, %[[DIV_ALIAS1:arg[0-9]+]]: i32) {
// IR: arith.muli %[[DIV_ALIAS0]], %[[DIV_ALIAS1]] : i32
// IR: candidate args(@arg0) (%[[DIV_INPUT_ALIAS:arg[0-9]+]]: i32) {
// IR: egraph.yield %[[DIV_INPUT_ALIAS]] : i32
// IR: eclass @__intern_divsi : i32 {
// IR: candidate args(@c2_i32, @c2_i32) (%[[INNER_DIV0:arg[0-9]+]]: i32, %[[INNER_DIV1:arg[0-9]+]]: i32) {
// IR: arith.divsi %[[INNER_DIV0]], %[[INNER_DIV1]] : i32
// IR: candidate args() {
// IR: arith.constant 1 : i32
// IR-NOT: eclass @__intern_muli : i32 {
// IR: return @divsi : i32
