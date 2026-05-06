#ifndef MLIR_EGRAPH_TEST_EGRAPHTEST_OPS
#define MLIR_EGRAPH_TEST_EGRAPHTEST_OPS

include "EGraphTest/IR/EGraphTestDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def EGraphTest_LeafOp : EGraphTest_Op<"leaf", [Pure]> {
  let summary = "Leaf value for egraph regression coverage";
  let description = [{
    Creates a value without operands so lit tests can build standalone match
    shapes inside a function body.
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def EGraphTest_OpAOp : EGraphTest_Op<"op_a", [Pure]> {
  let summary = "Binary test operation for egraph coverage";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, OptionalAttr<StrAttr>:$mode);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def EGraphTest_OpBOp : EGraphTest_Op<"op_b", [Pure]> {
  let summary = "Unary test operation for egraph coverage";
  let arguments = (ins AnyType:$input, OptionalAttr<StrAttr>:$tag);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$input attr-dict `:` functional-type(operands, results)";
  let extraClassDeclaration = [{
    ::llvm::LogicalResult fold(FoldAdaptor adaptor,
                               ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &
                                   results);
  }];
}

def EGraphTest_SplitOp : EGraphTest_Op<"split", [Pure]> {
  let summary = "Multi-result test operation for egraph coverage";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$first, AnyType:$second);
  let assemblyFormat =
      "$input attr-dict `:` functional-type(operands, results)";
}

def EGraphTest_Slice0Op : EGraphTest_Op<"slice0", [Pure]> {
  let summary = "First slice test operation for egraph coverage";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$input attr-dict `:` functional-type(operands, results)";
}

def EGraphTest_Slice1Op : EGraphTest_Op<"slice1", [Pure]> {
  let summary = "Second slice test operation for egraph coverage";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$input attr-dict `:` functional-type(operands, results)";
}

def EGraphTest_ImpureOp : EGraphTest_Op<"impure", [MemoryEffects<[MemWrite]>]> {
  let summary = "Side-effecting test operation for negative coverage";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$input attr-dict `:` functional-type(operands, results)";
}

#endif // MLIR_EGRAPH_TEST_EGRAPHTEST_OPS
