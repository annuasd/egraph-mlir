// RUN: mlir-egraph-opt %s -split-input-file -verify-diagnostics

// Return target checks cover missing symbols and result type mismatches.
module {
  egraph.egraph @bad_missing() -> i32 {
    egraph.return @missing : i32 // expected-error {{'egraph.return' op return target #0 references undefined symbol @missing}}
  }
}

// -----

module {
  egraph.egraph @bad_type() -> i32 {
    egraph.return @missing : f32 // expected-error {{'egraph.return' op return type #0 'f32' must match enclosing egraph result type 'i32'}}
  }
}

// -----

// Input checks keep payload typing and entry-block ownership explicit.
module {
  egraph.egraph @bad_input_type(%x: i32) {
    "egraph.input"(%x) <{sym_name = "x", payload_type = f32}> : (i32) -> () // expected-error {{'egraph.input' op payload type 'f32' must match bound value type 'i32'}}
    egraph.return
  }
}

// -----

module {
  egraph.egraph @bad_input_source(%x: i32) {
    %cast = builtin.unrealized_conversion_cast %x : i32 to i32
    egraph.input @y = %cast : i32 // expected-error {{'egraph.input' op bound value must be an argument of the enclosing egraph entry block}}
    egraph.return
  }
}

// -----

module {
  egraph.egraph @old_v1_eclass(%arg0: !egraph.evalue<i32>) { // expected-error {{egraph dialect does not define custom types}}
    %ev = egraph.eclass(%arg0) : (!egraph.evalue<i32>) -> !egraph.evalue<i32> {
    }
    egraph.return
  }
}

// -----

// EClass checks cover symbolic child refs, candidate argument typing, and yield typing.
module {
  egraph.egraph @bad_eclass_child(%x: i32) {
    egraph.input @x = %x : i32
    egraph.eclass @bad : i32 { // expected-error {{'egraph.eclass' op candidate region #0 child symbol #0 references undefined symbol @missing}}
      candidate args(@missing) (%a: i32) {
        egraph.yield %a : i32
      }
    }
    egraph.return
  }
}

// -----

module {
  egraph.egraph @bad_eclass_arg_type(%x: i32) {
    egraph.input @x = %x : i32
    egraph.eclass @bad : i64 { // expected-error {{'egraph.eclass' op candidate region #0 argument #0 type 'i64' must match child symbol @x payload type 'i32'}}
      candidate args(@x) (%a: i64) {
        egraph.yield %a : i64
      }
    }
    egraph.return
  }
}

// -----

module {
  egraph.egraph @bad_eclass_yield_type(%x: i32) {
    egraph.input @x = %x : i32
    egraph.eclass @bad : f32 {
      candidate args(@x) (%a: i32) {
        egraph.yield %a : i32 // expected-error {{'egraph.yield' op operand #0 type 'i32' must match parent eclass payload type 'f32'}}
      }
    }
    egraph.return
  }
}

// -----

// Symbol-table verification still rejects duplicate eclass symbols inside one egraph.
module {
  egraph.egraph @bad_duplicate_symbol(%x: i32) {
    egraph.input @x = %x : i32
    egraph.eclass @dup : i32 { // expected-note {{see existing symbol definition here}}
      candidate args(@x) (%a: i32) {
        egraph.yield %a : i32
      }
    }
    egraph.eclass @dup : i32 { // expected-error {{redefinition of symbol named 'dup'}}
      candidate args(@x) (%a: i32) {
        egraph.yield %a : i32
      }
    }
    egraph.return
  }
}
