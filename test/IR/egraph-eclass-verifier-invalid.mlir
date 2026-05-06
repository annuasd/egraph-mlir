// RUN: mlir-egraph-opt %s -split-input-file -verify-diagnostics

module {
  // Missing child symbols should fail before candidate parsing completes.
  egraph.egraph @bad_missing_child(%x: i32) {
    egraph.input @x = %x : i32
    egraph.eclass @bad : i32 { // expected-error {{'egraph.eclass' op candidate region #0 child symbol #0 references undefined symbol @missing}}
      candidate args(@missing) (%value: i32) {
        egraph.yield %value : i32
      }
    }
    egraph.return
  }
}

// -----

module {
  // Child payload and candidate argument types must match exactly.
  egraph.egraph @bad_child_arg_type(%x: i32) {
    egraph.input @x = %x : i32
    egraph.eclass @bad : f32 { // expected-error {{'egraph.eclass' op candidate region #0 argument #0 type 'f32' must match child symbol @x payload type 'i32'}}
      candidate args(@x) (%value: f32) {
        egraph.yield %value : f32
      }
    }
    egraph.return
  }
}

// -----

module {
  // Symbolic eclass results still enforce single-result yield arity.
  egraph.egraph @bad_yield_count(%x: i32) {
    egraph.input @x = %x : i32
    egraph.eclass @bad : i32 {
      candidate args(@x) (%value: i32) {
        egraph.yield %value, %value : i32, i32 // expected-error {{'egraph.yield' op operand count 2 must match symbolic eclass single-result payload count 1}}
      }
    }
    egraph.return
  }
}

// -----

module {
  // Yield operand types must match the parent eclass payload type.
  egraph.egraph @bad_yield_type(%x: i32) {
    egraph.input @x = %x : i32
    egraph.eclass @bad : f32 {
      candidate args(@x) (%value: i32) {
        egraph.yield %value : i32 // expected-error {{'egraph.yield' op operand #0 type 'i32' must match parent eclass payload type 'f32'}}
      }
    }
    egraph.return
  }
}

// -----

module {
  // Duplicate symbols should be diagnosed on the second definition.
  egraph.egraph @bad_duplicate_symbol(%x: i32) {
    egraph.input @x = %x : i32
    egraph.eclass @dup : i32 { // expected-note {{see existing symbol definition here}}
      candidate args(@x) (%value: i32) {
        egraph.yield %value : i32
      }
    }
    egraph.eclass @dup : i32 { // expected-error {{redefinition of symbol named 'dup'}}
      candidate args(@x) (%value: i32) {
        egraph.yield %value : i32
      }
    }
    egraph.return
  }
}

// -----

module {
  // Return types must stay consistent with the enclosing egraph signature.
  egraph.egraph @bad_return_type(%x: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.return @x : f32 // expected-error {{'egraph.return' op return type #0 'f32' must match enclosing egraph result type 'i32'}}
  }
}

// -----

module {
  // Generic assembly keeps the payload-type verifier path visible here.
  egraph.egraph @bad_input_type(%x: i32) {
    "egraph.input"(%x) <{sym_name = "x", payload_type = f32}> : (i32) -> () // expected-error {{'egraph.input' op payload type 'f32' must match bound value type 'i32'}}
    egraph.return
  }
}
