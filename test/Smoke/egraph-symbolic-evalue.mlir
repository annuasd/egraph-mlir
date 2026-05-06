// RUN: mlir-egraph-opt %s --test-egraph-symbolic-evalue -split-input-file 2>&1 | FileCheck %s

// Cover direct EValue and EOpRef lookup on one symbolic candidate.
module {
  egraph.egraph @symbolic_lookup(%x: i32) -> i32 {
    egraph.input @x = %x : i32

    egraph.eclass @c2 : i32 {
      candidate args() {
        %c2 = arith.constant 2 : i32
        egraph.yield %c2 : i32
      }
    }

    egraph.eclass @double : i32 {
      candidate args(@x, @c2) (%lhs: i32, %rhs: i32) {
        %mul = arith.muli %lhs, %rhs : i32
        egraph.yield %mul : i32
      }
    }

    return @double : i32
  }
}

// The first check group stays on the single candidate before any union-based query.
// CHECK: remark: constructed input EValue @x result #0 : 'i32'
// CHECK: remark: lookupValue(entry arg) -> @x
// CHECK: remark: symbolic EValue result slot #1 is reserved for future multi-result support
// CHECK: remark: constructed eclass EValue @c2 result #0 : 'i32'
// CHECK: remark: lookupValue(candidate arg #0) -> @x
// CHECK: remark: lookupValue(candidate arg #1) -> @c2
// CHECK: remark: lookupValue(yielded payload) -> @double
// CHECK: remark: lookupOpRef(candidate root) -> arith.muli
// CHECK: remark: EOpRef operand #0 -> @x
// CHECK: remark: EOpRef operand #1 -> @c2
// CHECK: remark: EOpRef result #0 -> @double result #0

// -----

// Cover EValue def queries after two eclasses become equivalent through union.
module {
  egraph.egraph @symbolic_query(%x: i32, %y: i32) -> i32 {
    egraph.input @x = %x : i32
    egraph.input @y = %y : i32

    egraph.eclass @lhs : i32 {
      candidate args(@x, @y) (%lhs: i32, %rhs: i32) {
        %result = egraph_test.op_a %lhs, %rhs {mode = "fast"} : (i32, i32) -> i32
        egraph.yield %result : i32
      }

      candidate args(@y, @x) (%rhs: i32, %lhs: i32) {
        %result = egraph_test.op_a %rhs, %lhs {mode = "slow"} : (i32, i32) -> i32
        egraph.yield %result : i32
      }
    }

    egraph.eclass @rhs : i32 {
      candidate args(@x) (%input: i32) {
        %result = egraph_test.op_b %input {tag = "equivalent"} : (i32) -> i32
        egraph.yield %result : i32
      }
    }

    return @rhs : i32
  }
}

// The second check group exercises leader-following, typed filtering and callback-based queries.
// CHECK-DAG: remark: EValue getDefs returned no candidate roots for @x
// CHECK-DAG: remark: EValue query followed leader @lhs
// CHECK-DAG: remark: EValue getDefs found 3 candidate root(s) through the leader
// CHECK-DAG: remark: EValue typed getDefs found 2 op_a candidate(s)
// CHECK-DAG: remark: EValue hasDef observed op_a and op_b defs after union
// CHECK-DAG: remark: EValue getUniqueDef returned the unique op_b candidate
// CHECK-DAG: remark: EValue getUniqueDef rejected multiple op_a candidates
// CHECK-DAG: remark: EValue matchDef accepted LogicalResult callback
// CHECK-DAG: remark: EValue matchDef accepted bool callback
// CHECK-DAG: remark: EValue matchDef rejected all-false bool callback
// CHECK-DAG: remark: EValue matchDef visited 2 op_a candidate(s) with a void callback
