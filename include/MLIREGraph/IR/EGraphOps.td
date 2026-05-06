#ifndef MLIR_EGRAPH_OPS
#define MLIR_EGRAPH_OPS

include "MLIREGraph/IR/EGraphDialect.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def EGraph_EGraphOp
    : EGraph_Op<
          "egraph", [FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface,
                     SingleBlockImplicitTerminator<"ReturnOp">, SymbolTable]> {
  let summary = "persistent egraph container";
  let description = [{
    `egraph.egraph` is the top-level persistent container for a single egraph.
    It uses function-like syntax: block arguments carry payload boundary
    values, and `egraph.return` names the e-class symbols that define the graph
    results.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<StrAttr>:$sym_visibility,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>];

  let extraClassDeclaration = [{
    static EGraphOp create(Location location, StringRef name, FunctionType type,
                           ArrayRef<NamedAttribute> attrs = {});
    LogicalResult verifyBody();

    Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    static StringRef getDefaultDialect() { return "egraph"; }

    bool isDeclaration() { return isExternal(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def EGraph_InputOp : EGraph_Op<"input", [HasParent<"EGraphOp">, Pure, Symbol]> {
  let summary = "egraph boundary input";
  let description = [{
    `egraph.input` binds a named e-class identity to a payload boundary value
    entering the enclosing `egraph.egraph`.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, AnyType:$value,
      TypeAttr:$payload_type);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def EGraph_ReturnOp
    : EGraph_Op<"return", [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                           HasParent<"EGraphOp">, Pure, ReturnLike,
                           Terminator]> {
  let summary = "egraph result terminator";
  let description = [{
    Terminates an `egraph.egraph` body and returns the referenced e-class
    symbols as the graph outputs.
  }];

  let arguments = (ins FlatSymbolRefArrayAttr:$targets, TypeArrayAttr:$types);
  let hasCustomAssemblyFormat = 1;
}

def EGraph_EClassOp
    : EGraph_Op<"eclass", [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                           HasParent<"EGraphOp">, Pure, Symbol]> {
  let summary = "persistent egraph equivalence class";
  let description = [{
    `egraph.eclass` declares a named e-class identity. Its payload type is
    stored on the symbol op, and each candidate region stores child e-class
    references as `FlatSymbolRefAttr` entries printed by `candidate args(@...)`.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$payload_type,
      ArrayAttr:$candidate_refs);
  let regions = (region VariadicRegion<SizedRegion<1>>:$candidates);
  let extraClassDeclaration = [{
    static StringRef getDefaultDialect() { return "egraph"; }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def EGraph_YieldOp : EGraph_Op<"yield", [HasParent<"EClassOp">, Pure,
                                         ReturnLike, Terminator]> {
  let summary = "candidate terminator";
  let description = [{
    Terminates an `egraph.eclass` candidate region and yields the payload value
    for that candidate.
  }];

  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "$values attr-dict `:` type($values)";
}

#endif // MLIR_EGRAPH_OPS
