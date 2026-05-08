// RUN: mlir-egraph-opt %s --test-egraph-worklist-driver 2>&1 | FileCheck %s --check-prefix=REMARK
// RUN: mlir-egraph-opt %s --test-egraph-worklist-driver 2>/dev/null | FileCheck %s --check-prefix=IR

module {
  // Exercise the driver against a clean symbolic snapshot with one rewritten root,
  // one no-op success, and one stale candidate left in the worklist.
  egraph.egraph @symbolic_worklist_driver(%x: i32, %y: i32, %z: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.input @y = %y : i32
    egraph.input @z = %z : i32

    // The first candidate is rewritten through a child-driven pattern.
    egraph.eclass @lhs : i32 {
      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        %result = egraph_test.op_a %lhs, %rhs {mode = "driver_child"} : (i32, i32) -> i32
        egraph.yield %result : i32
      }
    }

    // The second candidate only exercises the no-op success path.
    egraph.eclass @rhs : i32 {
      candidate args(@z) (%input: i32) {
        %result = egraph_test.op_b %input {tag = "driver_rhs"} : (i32) -> i32
        egraph.yield %result : i32
      }
    }

    // The parent observes the rewritten child leader after rebuild.
    egraph.eclass @parent : i32 {
      candidate args(@rhs) (%input: i32) {
        %result = egraph_test.op_b %input {tag = "driver_parent"} : (i32) -> i32
        egraph.yield %result : i32
      }
    }

    // The tail stays in the worklist long enough to become stale.
    egraph.eclass @tail : i32 {
      candidate args(@z) (%input: i32) {
        %result = egraph_test.op_b %input {tag = "driver_tail"} : (i32) -> i32
        egraph.yield %result : i32
      }
    }

    return @parent : i32
  }
}

// REMARK-DAG: remark: worklist driver discarded failed pattern
// REMARK-DAG: remark: worklist driver skipped rebuild for no-op success
// REMARK-DAG: remark: worklist driver rebuilt after batched commits
// REMARK-DAG: remark: worklist driver processed enqueued new candidate root
// REMARK-DAG: remark: worklist driver stats: limit=none limit_reached=false iterations={{[0-9]+}} enqueued_candidates={{[0-9]+}} skipped_candidate_cap=0 skipped_stale_refs={{[0-9]+}} matched_patterns={{[0-9]+}} changed_commits=1 rebuilds=2 changed=true

// IR-LABEL: module {
// IR: egraph.egraph @symbolic_worklist_driver(%[[X:arg[0-9]+]]: i32, %[[Y:arg[0-9]+]]: i32, %[[Z:arg[0-9]+]]: i32) -> i32 {
// IR: input @x = %[[X]] : i32
// IR: input @y = %[[Y]] : i32
// IR: input @z = %[[Z]] : i32
// The rewritten leader keeps its own candidate and absorbs the added alias path.
// IR: eclass @lhs : i32 {
// IR: candidate args(@x, @y) (%[[LHS_ARG0:arg[0-9]+]]: i32, %[[LHS_ARG1:arg[0-9]+]]: i32) {
// IR: egraph_test.op_a %[[LHS_ARG0]], %[[LHS_ARG1]] {mode = "driver_child"} : (i32, i32) -> i32
// IR: candidate args(@z) (%[[LHS_RHS_ARG:arg[0-9]+]]: i32) {
// IR: egraph_test.op_b %[[LHS_RHS_ARG]] {tag = "driver_rhs"} : (i32) -> i32
// IR-NOT: eclass @rhs : i32 {
// The parent rebinds to the rewritten leader after rebuild.
// IR: eclass @parent : i32 {
// IR: candidate args(@lhs) (%[[PARENT_ARG:arg[0-9]+]]: i32) {
// IR: egraph_test.op_b %[[PARENT_ARG]] {tag = "driver_parent"} : (i32) -> i32
// IR: candidate args(@x) (%[[PARENT_ALIAS_ARG:arg[0-9]+]]: i32) {
// IR: egraph_test.op_b %[[PARENT_ALIAS_ARG]] {tag = "driver_added"} : (i32) -> i32
// IR-NOT: eclass @__intern_op_b : i32 {
// The tail remains as the final live root after stale refs are skipped.
// IR: eclass @tail : i32 {
// IR: egraph_test.op_b {{.*}} {tag = "driver_tail"} : (i32) -> i32
// IR: return @parent : i32
