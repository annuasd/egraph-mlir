// RUN: mlir-egraph-opt %s --test-egraph-scratch-rewriter 2>&1 | FileCheck %s --check-prefix=SCRATCH
// RUN: mlir-egraph-opt %s --test-egraph-rewrite-events 2>&1 | FileCheck %s --check-prefix=EVENT
// RUN: mlir-egraph-opt %s --test-egraph-event-validation 2>&1 | FileCheck %s --check-prefix=VALID

// One symbolic egraph fixture drives three test-only passes so the remarks can
// isolate scratch lifetime, event recording, and validation behavior.

module {
  func.func @external_leaf_holder() {
    %external = egraph_test.leaf {name = "external"} : i32
    func.return
  }

  egraph.egraph @rewrite_events(%x: i32, %y: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.input @y = %y : i32

    egraph.eclass @value : i32 {
      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        %result = egraph_test.op_a %lhs, %rhs {mode = "validation_replacement"} : (i32, i32) -> i32
        egraph.yield %result : i32
      }

      candidate args(@x) (%input: i32) {
        %result = egraph_test.op_b %input {tag = "validation_equivalence"} : (i32) -> i32
        egraph.yield %result : i32
      }
    }

    return @value : i32
  }
}

// Scratch-only paths must discard detached DAG state whenever no event commits.
// SCRATCH-DAG: remark: scratch op created in transaction-owned detached block
// SCRATCH-DAG: remark: pattern failure discarded scratch block
// SCRATCH-DAG: remark: pattern failure left persistent egraph unchanged
// SCRATCH-DAG: remark: success without events discarded scratch block
// SCRATCH-DAG: remark: success without events left persistent egraph unchanged

// Rewrite-event recording accepts legal roots/targets and rejects scratch oldOp.
// EVENT-DAG: remark: replacement event recorded from EOpRef
// EVENT-DAG: remark: replacement event recorded from Operation lookup
// EVENT-DAG: remark: scratch oldOp replacement rejected
// EVENT-DAG: remark: equivalence event recorded from EOpRef
// EVENT-DAG: remark: equivalence event recorded with scratch target
// EVENT-DAG: remark: event recording left persistent egraph unchanged

// Validation distinguishes legal scratch/egraph-owned values from external uses.
// VALID-DAG: remark: event validation accepted scratch replacement value
// VALID-DAG: remark: event validation accepted egraph-owned replacement value
// VALID-DAG: remark: event validation accepted legal alias replacement value
// VALID-DAG: remark: event validation accepted egraph-owned equivalence target
// VALID-DAG: remark: event validation accepted scratch equivalence target
// VALID-DAG: remark: event validation rejected replacement result count mismatch
// VALID-DAG: remark: event validation rejected replacement type mismatch
// VALID-DAG: remark: event validation rejected illegal external replacement value
// VALID-DAG: remark: illegal external equivalence target rejected before validation
// VALID-DAG: remark: scratch oldOp replacement rejected before validation
