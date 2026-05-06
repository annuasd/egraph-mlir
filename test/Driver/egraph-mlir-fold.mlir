// RUN: mlir-egraph-opt %s --test-egraph-mlir-fold 2>&1 | FileCheck %s --check-prefix=REMARK
// RUN: mlir-egraph-opt %s --test-egraph-mlir-fold 2>/dev/null | FileCheck %s --check-prefix=IR

module {
  func.func @fold_demo() -> i32 {
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %sum = arith.addi %c2, %c3 : i32
    return %sum : i32
  }

  func.func @fold_value_demo(%x: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %sum = arith.addi %x, %c0 : i32
    return %sum : i32
  }

  func.func @fold_inplace_demo(%x: i32) -> i32 {
    %tagged = egraph_test.op_b %x {tag = "strip"} : (i32) -> i32
    return %tagged : i32
  }
}

// REMARK: remark: egraph fold stayed disabled by default in @addi
// REMARK: remark: egraph fold preserved the original addi candidate in @addi
// REMARK: remark: egraph fold materialized delayed constant in @addi
// REMARK: remark: egraph fold skipped conflicting user rewrite in @addi
// REMARK: remark: egraph fold materialized delayed value alias in @addi
// REMARK: remark: egraph fold preserved the original tagged op_b candidate in @op_b
// REMARK: remark: egraph fold materialized an in-place tagless op_b candidate in @op_b

// IR-LABEL: module {
// IR: egraph.egraph @fold_demo_egraph() -> i32 {
// IR-NOT: evalue<
// IR: eclass @c2_i32 : i32 {
// IR: arith.constant 2 : i32
// IR: eclass @c3_i32 : i32 {
// IR: arith.constant 3 : i32
// IR: eclass @addi : i32 {
// IR: candidate args(@c2_i32, @c3_i32) (%[[ADD_LHS:arg[0-9]+]]: i32, %[[ADD_RHS:arg[0-9]+]]: i32) {
// IR: arith.addi %[[ADD_LHS]], %[[ADD_RHS]] : i32
// IR: candidate args() {
// IR: arith.constant 5 : i32
// IR: return @addi : i32
// IR: egraph.egraph @fold_value_demo_egraph(%[[INPUT:arg[0-9]+]]: i32) -> i32 {
// IR: input @arg0 = %[[INPUT]] : i32
// IR: eclass @c0_i32 : i32 {
// IR: arith.constant 0 : i32
// IR: eclass @addi : i32 {
// IR: candidate args(@{{(addi|arg[0-9]+)}}, @c0_i32) (%[[VALUE_ADD_LHS:arg[0-9]+]]: i32, %[[VALUE_ADD_RHS:arg[0-9]+]]: i32) {
// IR: arith.addi %[[VALUE_ADD_LHS]], %[[VALUE_ADD_RHS]] : i32
// IR: candidate args(@arg0) (%[[ALIAS:arg[0-9]+]]: i32) {
// IR: egraph.yield %[[ALIAS]] : i32
// IR: return @addi : i32
// IR: egraph.egraph @fold_inplace_demo_egraph(%[[INPLACE_INPUT:arg[0-9]+]]: i32) -> i32 {
// IR: input @arg0 = %[[INPLACE_INPUT]] : i32
// IR: eclass @op_b : i32 {
// IR-DAG: egraph_test.op_b %{{.*}} {tag = "strip"} : (i32) -> i32
// IR-DAG: egraph_test.op_b %{{.*}} : (i32) -> i32
// IR: return @op_b : i32
