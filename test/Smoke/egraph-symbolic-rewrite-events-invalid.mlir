// RUN: mlir-egraph-opt %s --test-egraph-negative-failures -split-input-file -verify-diagnostics

// Each split-input module isolates one rewrite-event failure mode so the
// expected diagnostic stays anchored on the candidate under test.

// Replacement validation rejects payload type mismatches.
module {
  func.func @reject_transaction_type_mismatch() {
    func.return
  }

  egraph.egraph @rewrite_events(%x: i32, %y: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.input @y = %y : i32

    egraph.eclass @value : i32 {
      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        // expected-error @+1 {{negative test rejected replacement type mismatch}}
        %result = egraph_test.op_a %lhs, %rhs {mode = "negative_type_mismatch"} : (i32, i32) -> i32
        egraph.yield %result : i32
      }

      candidate args(@x) (%input: i32) {
        %result = egraph_test.op_b %input {tag = "negative_equivalence"} : (i32) -> i32
        egraph.yield %result : i32
      }
    }

    return @value : i32
  }
}

// -----

// Replacement events cannot capture values defined outside the transaction.
module {
  func.func @reject_illegal_external_value() {
    %external = egraph_test.leaf {name = "external"} : i32
    func.return
  }

  egraph.egraph @rewrite_events(%x: i32, %y: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.input @y = %y : i32

    egraph.eclass @value : i32 {
      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        // expected-error @+1 {{negative test rejected illegal external replacement value}}
        %result = egraph_test.op_a %lhs, %rhs {mode = "negative_external"} : (i32, i32) -> i32
        egraph.yield %result : i32
      }

      candidate args(@x) (%input: i32) {
        %result = egraph_test.op_b %input {tag = "negative_equivalence"} : (i32) -> i32
        egraph.yield %result : i32
      }
    }

    return @value : i32
  }
}

// -----

// Replacement events must use a live egraph-owned old root, not a scratch op.
module {
  func.func @reject_scratch_old_op() {
    func.return
  }

  egraph.egraph @rewrite_events(%x: i32, %y: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.input @y = %y : i32

    egraph.eclass @value : i32 {
      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        // expected-error @+1 {{negative test rejected scratch oldOp replacement}}
        %result = egraph_test.op_a %lhs, %rhs {mode = "negative_scratch_old"} : (i32, i32) -> i32
        egraph.yield %result : i32
      }

      candidate args(@x) (%input: i32) {
        %result = egraph_test.op_b %input {tag = "negative_equivalence"} : (i32) -> i32
        egraph.yield %result : i32
      }
    }

    return @value : i32
  }
}
