#include "MLIREGraph/IR/EGraphOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>

using namespace mlir;
using namespace mlir::egraph;

namespace {
LogicalResult
readCandidateSymbolRefs(EClassOp op, Attribute attr, unsigned candidateOrdinal,
                        SmallVectorImpl<FlatSymbolRefAttr> &refs) {
  auto array = dyn_cast<ArrayAttr>(attr);
  if (!array)
    return op.emitOpError("candidate_refs entry #")
           << candidateOrdinal << " must be a symbol ref array attribute";

  refs.clear();
  refs.reserve(array.size());
  for (Attribute element : array) {
    auto symbol = dyn_cast<FlatSymbolRefAttr>(element);
    if (!symbol)
      return op.emitOpError("candidate_refs entry #")
             << candidateOrdinal << " must be a symbol ref array attribute";

    refs.push_back(symbol);
  }

  return success();
}

Type getStoredType(Attribute attr) {
  auto typeAttr = dyn_cast<TypeAttr>(attr);
  return typeAttr ? typeAttr.getValue() : Type();
}

Type getPayloadTypeFromSymbol(Operation *symbolOp) {
  if (auto input = dyn_cast_or_null<InputOp>(symbolOp))
    return input.getPayloadType();
  if (auto eclass = dyn_cast_or_null<EClassOp>(symbolOp))
    return eclass.getPayloadType();
  return Type();
}

ParseResult parseCandidateSymbolRefs(OpAsmParser &parser,
                                     SmallVectorImpl<Attribute> &refs) {
  if (parser.parseKeyword("args"))
    return failure();

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        [&]() -> ParseResult {
                                          FlatSymbolRefAttr ref;
                                          if (parser.parseAttribute(ref))
                                            return failure();

                                          refs.push_back(ref);
                                          return success();
                                        });
}

ParseResult parseReturnTargets(OpAsmParser &parser,
                               SmallVectorImpl<Attribute> &targets) {
  FlatSymbolRefAttr target;
  OptionalParseResult parsedTarget = parser.parseOptionalAttribute(target);
  if (!parsedTarget.has_value())
    return success();
  if (failed(*parsedTarget))
    return failure();

  targets.push_back(target);
  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseAttribute(target))
      return failure();
    targets.push_back(target);
  }

  return success();
}
} // namespace

EGraphOp EGraphOp::create(Location location, StringRef name, FunctionType type,
                          ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  EGraphOp::build(builder, state, name, type, attrs, {});
  return cast<EGraphOp>(Operation::create(state));
}

void EGraphOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                     FunctionType type, ArrayRef<NamedAttribute> attrs,
                     ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;

  assert(type.getNumInputs() == argAttrs.size() &&
         "entry block attribute count must match function inputs");
  call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/{},
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

ParseResult EGraphOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();

  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  bool isVariadic = false;
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<Type> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  if (function_interface_impl::parseFunctionSignatureWithArguments(
          parser, /*allowVariadic=*/false, entryArgs, isVariadic, resultTypes,
          resultAttrs))
    return failure();

  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (const OpAsmParser::Argument &arg : entryArgs)
    argTypes.push_back(arg.type);

  result.addAttribute(
      getFunctionTypeAttrName(result.name),
      TypeAttr::get(builder.getFunctionType(argTypes, resultTypes)));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  call_interface_impl::addArgAndResultAttrs(
      builder, result, entryArgs, resultAttrs, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs))
    return failure();

  return success();
}

void EGraphOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getSymName());
  function_interface_impl::printFunctionSignature(
      printer, *this, getArgumentTypes(), /*isVariadic=*/false,
      getResultTypes());
  function_interface_impl::printFunctionAttributes(printer, *this,
                                                   {getFunctionTypeAttrName(),
                                                    getArgAttrsAttrName(),
                                                    getResAttrsAttrName()});
  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

LogicalResult EGraphOp::verifyBody() {
  if (isExternal())
    return emitOpError("requires a body");

  Block &entryBlock = getBody().front();
  unsigned numArguments = getNumArguments();
  if (entryBlock.getNumArguments() != numArguments)
    return emitOpError("entry block must have ")
           << numArguments << " arguments to match egraph signature";

  for (auto [index, expectedType, actualType] :
       llvm::enumerate(getArgumentTypes(), entryBlock.getArgumentTypes())) {
    if (expectedType != actualType) {
      return emitOpError("type of entry block argument #")
             << index << '(' << actualType
             << ") must match the type of the corresponding argument in "
             << "egraph signature(" << expectedType << ')';
    }
  }

  return success();
}

ParseResult InputOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseEqual())
    return failure();

  OpAsmParser::UnresolvedOperand value;
  if (parser.parseOperand(value) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (result.attributes.get("payload_type"))
    return parser.emitError(parser.getCurrentLocation())
           << "payload_type is printed with the custom input syntax";

  Type payloadType;
  if (parser.parseColonType(payloadType) ||
      parser.resolveOperand(value, payloadType, result.operands))
    return failure();

  result.addAttribute("payload_type", TypeAttr::get(payloadType));
  return success();
}

void InputOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getSymName());
  printer << " = " << getValue();
  printer.printOptionalAttrDict(
      (*this)->getAttrs(),
      {mlir::SymbolTable::getSymbolAttrName(), "payload_type"});
  printer << " : ";
  printer.printType(getPayloadType());
}

LogicalResult InputOp::verify() {
  Type payloadType = getPayloadType();
  Type valueType = getValue().getType();
  if (payloadType != valueType)
    return emitOpError("payload type ")
           << payloadType << " must match bound value type " << valueType;

  auto egraph = cast<EGraphOp>((*this)->getParentOp());
  if (egraph.getBody().empty())
    return emitOpError("requires a non-empty enclosing egraph body");

  auto blockArg = dyn_cast<BlockArgument>(getValue());
  if (!blockArg || blockArg.getOwner() != &egraph.getBody().front())
    return emitOpError(
        "bound value must be an argument of the enclosing egraph entry block");

  return success();
}

ParseResult ReturnOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  SmallVector<Attribute> targets;
  SmallVector<Type> types;

  if (parseReturnTargets(parser, targets))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (result.attributes.get("targets") || result.attributes.get("types"))
    return parser.emitError(parser.getCurrentLocation())
           << "targets and types are printed with the custom return syntax";

  if (!targets.empty() && (parser.parseColon() || parser.parseTypeList(types)))
    return failure();

  result.addAttribute("targets", builder.getArrayAttr(targets));
  result.addAttribute("types", builder.getTypeArrayAttr(types));
  return success();
}

void ReturnOp::print(OpAsmPrinter &printer) {
  ArrayAttr targets = getTargets();
  ArrayAttr types = getTypes();

  if (!targets.empty()) {
    printer << " ";
    llvm::interleaveComma(targets, printer.getStream(), [&](Attribute attr) {
      printer.printSymbolName(cast<FlatSymbolRefAttr>(attr).getValue());
    });
  }

  printer.printOptionalAttrDict((*this)->getAttrs(), {"targets", "types"});

  if (!types.empty()) {
    printer << " : ";
    llvm::interleaveComma(types, printer.getStream(), [&](Attribute attr) {
      printer.printType(cast<TypeAttr>(attr).getValue());
    });
  }
}

LogicalResult ReturnOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto egraph = cast<EGraphOp>((*this)->getParentOp());
  ArrayAttr targets = getTargets();
  ArrayAttr types = getTypes();

  if (targets.size() != types.size())
    return emitOpError("requires the same number of target symbols and types");

  if (targets.size() != egraph.getNumResults())
    return emitOpError("returns ")
           << targets.size() << " symbol(s) but enclosing egraph requires "
           << egraph.getNumResults() << " result(s)";

  for (auto indexedType : llvm::enumerate(types)) {
    unsigned index = indexedType.index();
    Type actualType = getStoredType(indexedType.value());
    Type expectedType = egraph.getResultTypes()[index];
    if (!actualType)
      return emitOpError("type #") << index << " must be a type attribute";
    if (actualType != expectedType)
      return emitOpError("return type #")
             << index << " " << actualType
             << " must match enclosing egraph result type " << expectedType;
  }

  for (auto indexedTarget : llvm::enumerate(targets)) {
    unsigned index = indexedTarget.index();
    auto target = cast<FlatSymbolRefAttr>(indexedTarget.value());
    Operation *symbolOp =
        symbolTable.lookupSymbolIn(egraph, SymbolRefAttr(target));
    if (!symbolOp)
      return emitOpError("return target #")
             << index << " references undefined symbol @" << target.getValue();

    Type targetType = getPayloadTypeFromSymbol(symbolOp);
    if (!targetType)
      return emitOpError("return target #")
             << index
             << " must reference a payload-bearing egraph symbol, but @"
             << target.getValue() << " does not";

    Type actualType = getStoredType(types[index]);
    if (actualType != targetType)
      return emitOpError("return type #")
             << index << " " << actualType << " must match target symbol @"
             << target.getValue() << " payload type " << targetType;
  }

  return success();
}

ParseResult EClassOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();

  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseColon())
    return failure();

  Type payloadType;
  if (parser.parseType(payloadType) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  if (result.attributes.get("payload_type") ||
      result.attributes.get("candidate_refs"))
    return parser.emitError(parser.getCurrentLocation())
           << "payload_type and candidate_refs are printed with the custom "
              "eclass syntax";

  // Candidate regions are parsed in order and matched to candidate_refs
  // entries positionally. The verifier keeps the arity and payload checks in
  // sync with this stored metadata.
  if (parser.parseLBrace())
    return failure();

  SmallVector<Attribute> candidateRefs;
  while (failed(parser.parseOptionalRBrace())) {
    if (parser.parseKeyword("candidate"))
      return failure();

    SmallVector<Attribute> refs;
    if (parseCandidateSymbolRefs(parser, refs))
      return failure();

    SmallVector<OpAsmParser::Argument> blockArgs;
    if (succeeded(parser.parseOptionalLParen())) {
      if (failed(parser.parseOptionalRParen())) {
        // Block arguments mirror the referenced child payloads one-for-one.
        do {
          OpAsmParser::Argument arg;
          if (parser.parseArgument(arg, /*allowType=*/true))
            return failure();
          blockArgs.push_back(arg);
        } while (succeeded(parser.parseOptionalComma()));
        if (parser.parseRParen())
          return failure();
      }
    }

    candidateRefs.push_back(builder.getArrayAttr(refs));

    std::unique_ptr<Region> candidate = std::make_unique<Region>();
    if (parser.parseRegion(*candidate, blockArgs))
      return failure();

    result.addRegion(std::move(candidate));
  }

  result.addAttribute("payload_type", TypeAttr::get(payloadType));
  result.addAttribute("candidate_refs",
                      ArrayAttr::get(parser.getContext(), candidateRefs));
  return success();
}

void EClassOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getSymName());
  printer << " : ";
  printer.printType(getPayloadType());
  printer.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {mlir::SymbolTable::getSymbolAttrName(),
                            "payload_type", "candidate_refs"});
  printer << " {";
  printer.increaseIndent();

  ArrayAttr candidateRefs = getCandidateRefs();
  for (auto indexedRegion : llvm::enumerate(getCandidates())) {
    unsigned candidateOrdinal = indexedRegion.index();
    Region &candidate = indexedRegion.value();
    Block &block = candidate.front();

    SmallVector<FlatSymbolRefAttr> refs;
    assert(succeeded(readCandidateSymbolRefs(*this,
                                             candidateRefs[candidateOrdinal],
                                             candidateOrdinal, refs)) &&
           "invalid candidate_refs should be rejected by the verifier");

    printer.printNewline();
    printer << "candidate args(";
    llvm::interleaveComma(refs, printer.getStream(),
                          [&](FlatSymbolRefAttr ref) {
                            printer.printSymbolName(ref.getValue());
                          });
    printer << ")";
    if (block.getNumArguments() != 0) {
      printer << " (";
      llvm::interleaveComma(
          block.getArguments(), printer.getStream(),
          [&](BlockArgument arg) { printer.printRegionArgument(arg); });
      printer << ")";
    }
    printer << " ";
    printer.printRegion(candidate, /*printEntryBlockArgs=*/false);
  }

  printer.decreaseIndent();
  printer.printNewline();
  printer << "}";
}

LogicalResult EClassOp::verify() {
  ArrayAttr candidateRefs = getCandidateRefs();
  MutableArrayRef<Region> candidates = getCandidates();

  if (candidateRefs.size() != candidates.size())
    return emitOpError("requires candidate_refs to contain one entry per "
                       "candidate region, got ")
           << candidateRefs.size() << " entries for " << candidates.size()
           << " regions";

  for (auto indexedRegion : llvm::enumerate(candidates)) {
    unsigned candidateOrdinal = indexedRegion.index();
    Region &candidate = indexedRegion.value();

    if (candidate.empty() || !llvm::hasSingleElement(candidate))
      return emitOpError("candidate region #")
             << candidateOrdinal << " must contain exactly one block";

    SmallVector<FlatSymbolRefAttr> refs;
    if (failed(readCandidateSymbolRefs(*this, candidateRefs[candidateOrdinal],
                                       candidateOrdinal, refs)))
      return failure();

    Block &block = candidate.front();
    if (block.getNumArguments() != refs.size())
      return emitOpError("candidate region #")
             << candidateOrdinal << " has " << block.getNumArguments()
             << " block arguments but candidate_refs entry has " << refs.size()
             << " symbol refs";

    auto yield = dyn_cast_or_null<YieldOp>(block.getTerminator());
    if (!yield)
      return emitOpError("candidate region #")
             << candidateOrdinal << " must terminate with egraph.yield";

    // v1.1 symbolic eclasses are still single-result, so every candidate must
    // yield exactly one payload value with the enclosing payload type.
    if (yield.getNumOperands() != 1)
      return yield.emitOpError("operand count ")
             << yield.getNumOperands() << " must match symbolic eclass "
             << "single-result payload count 1";

    Type actualType = yield.getValues().front().getType();
    Type expectedType = getPayloadType();
    if (actualType != expectedType)
      return yield.emitOpError("operand #0 type ")
             << actualType << " must match parent eclass payload type "
             << expectedType;
  }

  return success();
}

LogicalResult EClassOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto egraph = cast<EGraphOp>((*this)->getParentOp());
  ArrayAttr candidateRefs = getCandidateRefs();
  MutableArrayRef<Region> candidates = getCandidates();

  if (candidateRefs.size() != candidates.size())
    return success();

  for (auto indexedRegion : llvm::enumerate(candidates)) {
    unsigned candidateOrdinal = indexedRegion.index();
    Region &candidate = indexedRegion.value();

    SmallVector<FlatSymbolRefAttr> refs;
    if (failed(readCandidateSymbolRefs(*this, candidateRefs[candidateOrdinal],
                                       candidateOrdinal, refs)))
      return failure();

    SmallVector<Type> refTypes;
    refTypes.reserve(refs.size());
    for (auto indexedRef : llvm::enumerate(refs)) {
      unsigned argIndex = indexedRef.index();
      FlatSymbolRefAttr ref = indexedRef.value();
      Operation *symbolOp = symbolTable.lookupSymbolIn(egraph, ref);
      if (!symbolOp)
        return emitOpError("candidate region #")
               << candidateOrdinal << " child symbol #" << argIndex
               << " references undefined symbol @" << ref.getValue();

      Type refType = getPayloadTypeFromSymbol(symbolOp);
      if (!refType)
        return emitOpError("candidate region #")
               << candidateOrdinal << " child symbol #" << argIndex
               << " must reference a payload-bearing egraph symbol, but @"
               << ref.getValue() << " does not";

      refTypes.push_back(refType);
    }

    if (candidate.empty() || !llvm::hasSingleElement(candidate))
      continue;

    Block &block = candidate.front();
    if (block.getNumArguments() != refs.size())
      continue;

    for (auto indexedArg : llvm::enumerate(block.getArguments())) {
      unsigned argIndex = indexedArg.index();
      Type actualType = indexedArg.value().getType();
      Type expectedType = refTypes[argIndex];
      if (actualType != expectedType)
        return emitOpError("candidate region #")
               << candidateOrdinal << " argument #" << argIndex << " type "
               << actualType << " must match child symbol @"
               << refs[argIndex].getValue() << " payload type " << expectedType;
    }
  }

  return success();
}

#define GET_OP_CLASSES
#include "MLIREGraph/IR/EGraphOps.cpp.inc"
