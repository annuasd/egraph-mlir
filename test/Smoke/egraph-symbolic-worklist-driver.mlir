// RUN: mlir-egraph-opt %s --test-egraph-worklist-driver 2>&1 | FileCheck %s
// Keep this coverage focused on worklist enqueue, skip, and rebuild paths.

module {
  egraph.egraph @symbolic_worklist_driver(%x: i32, %y: i32, %z: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.input @y = %y : i32
    egraph.input @z = %z : i32

    // This root exercises the nested match and enqueue path.
    egraph.eclass @lhs : i32 {
      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        %result = egraph_test.op_a %lhs, %rhs {mode = "driver_child"} : (i32, i32) -> i32
        egraph.yield %result : i32
      }
    }

    // This root exercises the rewritten candidate and parent tracking path.
    egraph.eclass @rhs : i32 {
      candidate args(@z) (%input: i32) {
        %result = egraph_test.op_b %input {tag = "driver_rhs"} : (i32) -> i32
        egraph.yield %result : i32
      }
    }

    // This root exercises rebuild visibility for a dependent parent.
    egraph.eclass @parent : i32 {
      candidate args(@rhs) (%input: i32) {
        %result = egraph_test.op_b %input {tag = "driver_parent"} : (i32) -> i32
        egraph.yield %result : i32
      }
    }

    // This root exercises stale tail candidate skipping.
    egraph.eclass @tail : i32 {
      candidate args(@z) (%input: i32) {
        %result = egraph_test.op_b %input {tag = "driver_tail"} : (i32) -> i32
        egraph.yield %result : i32
      }
    }

    return @parent : i32
  }
}

// CHECK-DAG: remark: worklist driver matched nested op_a
// CHECK-DAG: remark: worklist driver processed enqueued new candidate root
// CHECK-DAG: remark: worklist driver reached iteration limit
// CHECK-DAG: remark: worklist driver skipped per-eclass candidate cap
// CHECK-DAG: remark: worklist driver added MLIR fold alternative
// CHECK-DAG: remark: worklist driver rebuilt dirty graph before matching
// CHECK-DAG: remark: worklist driver added equivalent candidate
// CHECK-DAG: remark: worklist driver discarded failed pattern
// CHECK-DAG: remark: worklist driver skipped rebuild for no-op success
// CHECK-DAG: remark: worklist driver rebuilt after batched commits
// CHECK-DAG: remark: worklist driver stats counted matched patterns
// CHECK-DAG: remark: worklist driver stats: limit=none limit_reached=false iterations={{[0-9]+}} enqueued_candidates={{[0-9]+}} skipped_candidate_cap=0 skipped_stale_refs={{[0-9]+}} matched_patterns={{[0-9]+}} changed_commits=1 rebuilds=2 changed=true
