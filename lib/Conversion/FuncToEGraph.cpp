#include "MLIREGraph/EGraph/FuncToEGraph.h"

#include "MLIREGraph/IR/EGraphOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include <string>

using namespace mlir;
using namespace mlir::egraph;

namespace {
StringRef getLeafOperationName(Operation *op) {
  StringRef fullName = op->getName().getStringRef();
  auto split = fullName.rsplit('.');
  return split.second.empty() ? split.first : split.second;
}

std::string sanitizeSymbolStem(StringRef stem) {
  SmallString<64> sanitized;
  for (char ch : stem) {
    if (llvm::isAlnum(static_cast<unsigned char>(ch)) || ch == '_') {
      sanitized.push_back(ch);
      continue;
    }

    if (sanitized.empty() || sanitized.back() == '_')
      continue;
    sanitized.push_back('_');
  }

  while (!sanitized.empty() && sanitized.back() == '_')
    sanitized.pop_back();

  if (sanitized.empty())
    return "sym";
  if (llvm::isDigit(static_cast<unsigned char>(sanitized.front())))
    return (Twine("s") + sanitized).str();
  return sanitized.str().str();
}

std::string getAsmResultName(Operation *op, unsigned resultIndex) {
  if (resultIndex >= op->getNumResults())
    return {};

  std::string name;
  if (auto asmInterface = dyn_cast<OpAsmOpInterface>(op)) {
    asmInterface.getAsmResultNames([&](Value value, StringRef candidate) {
      if (value == op->getResult(resultIndex) && !candidate.empty())
        name = candidate.str();
    });
  }
  return name;
}

class DeterministicSymbolNamer {
public:
  explicit DeterministicSymbolNamer(Operation *symbolTableOp)
      : symbolTableOp(symbolTableOp) {}

  std::string claim(StringRef preferredStem) {
    std::string base = sanitizeSymbolStem(preferredStem);
    auto isTaken = [&](StringRef candidate) {
      return reservedNames.contains(candidate) ||
             SymbolTable::lookupSymbolIn(symbolTableOp, candidate);
    };

    std::string candidate = base;
    if (isTaken(candidate)) {
      unsigned &suffix = nextSuffixByStem[base];
      do {
        candidate = (Twine(base) + "_" + Twine(suffix++)).str();
      } while (isTaken(candidate));
    }

    reservedNames.insert(candidate);
    return candidate;
  }

  std::string claimEGraphName(func::FuncOp func) {
    return claim((func.getSymName() + "_egraph").str());
  }

  std::string claimInputName(unsigned argIndex) {
    return claim((Twine("arg") + Twine(argIndex)).str());
  }

  std::string claimValueName(Value value) {
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      return claimInputName(blockArg.getArgNumber());

    Operation *definingOp = value.getDefiningOp();
    assert(definingOp && "expected imported value to have a defining op");
    unsigned resultIndex = cast<OpResult>(value).getResultNumber();
    std::string stem = getAsmResultName(definingOp, resultIndex);
    if (stem.empty())
      stem = getLeafOperationName(definingOp).str();
    if (definingOp->getNumResults() > 1)
      stem = (Twine(stem) + "_" + Twine(resultIndex)).str();
    return claim(stem);
  }

private:
  Operation *symbolTableOp;
  llvm::StringSet<> reservedNames;
  llvm::StringMap<unsigned> nextSuffixByStem;
};

FailureOr<EClassOp>
importOperationToEClass(Operation *op, OpBuilder &builder,
                        DeterministicSymbolNamer &namer,
                        DenseMap<Value, FlatSymbolRefAttr> &importedSymbols) {
  if (op->getNumResults() != 1)
    return op->emitOpError(
        "func-to-egraph import currently supports only single-result pure "
        "operations");

  SmallVector<Attribute> childRefs;
  childRefs.reserve(op->getNumOperands());
  for (OpOperand &operand : op->getOpOperands()) {
    auto symbolIt = importedSymbols.find(operand.get());
    if (symbolIt == importedSymbols.end())
      return op->emitOpError("operand #")
             << operand.getOperandNumber()
             << " has not been imported into egraph";

    childRefs.push_back(symbolIt->second);
  }

  Value result = op->getResult(0);
  std::string eclassName = namer.claimValueName(result);
  auto nameAttr = builder.getStringAttr(eclassName);
  // The persistent identity is the e-class symbol; the cloned payload op stays
  // local to the candidate region and only contributes its yielded value.
  auto eclass = EClassOp::create(
      builder, op->getLoc(), nameAttr, TypeAttr::get(result.getType()),
      builder.getArrayAttr({builder.getArrayAttr(childRefs)}),
      /*candidatesCount=*/1);
  Region &candidate = eclass.getCandidates().front();
  Block *candidateBlock = new Block();
  candidate.push_back(candidateBlock);

  IRMapping mapping;
  for (OpOperand &operand : op->getOpOperands()) {
    BlockArgument candidateArg =
        candidateBlock->addArgument(operand.get().getType(), op->getLoc());
    mapping.map(operand.get(), candidateArg);
  }

  OpBuilder candidateBuilder = OpBuilder::atBlockBegin(candidateBlock);
  Operation *clonedOp = candidateBuilder.clone(*op, mapping);
  // Keep the imported candidate shape close to the original SSA payload
  // operation so later review can compare the clone against the source op.
  YieldOp::create(candidateBuilder, op->getLoc(), clonedOp->getResult(0));

  importedSymbols.try_emplace(result, FlatSymbolRefAttr::get(nameAttr));
  return eclass;
}

LogicalResult verifyOperandIsBlockLocal(Operation *op, OpOperand &operand,
                                        Block &block) {
  Value value = operand.get();
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() == &block)
      return success();

    return op->emitOpError("operand #")
           << operand.getOperandNumber()
           << " uses a block argument from another block; cross-block values "
              "are not supported by func-to-egraph import";
  }

  Operation *definingOp = value.getDefiningOp();
  if (definingOp && definingOp->getBlock() == &block)
    return success();

  return op->emitOpError("operand #")
         << operand.getOperandNumber()
         << " uses a value from another block; cross-block values are not "
            "supported by func-to-egraph import";
}

LogicalResult verifyResultUsersAreBlockLocal(Operation *op, Block &block) {
  for (auto indexedResult : llvm::enumerate(op->getResults())) {
    if (!indexedResult.value().isUsedOutsideOfBlock(&block))
      continue;

    return op->emitOpError("result #")
           << indexedResult.index()
           << " is used outside the defining block; cross-block values are not "
              "supported by func-to-egraph import";
  }

  return success();
}

LogicalResult verifyOperationIsEligibleForImport(Operation *op, Block &block) {
  if (isa<func::ReturnOp>(op))
    return success();

  if (op->getNumSuccessors() != 0 || isa<BranchOpInterface>(op) ||
      isa<RegionBranchOpInterface>(op))
    return op->emitOpError(
        "control-flow operation cannot be imported into egraph");

  if (isa<CallOpInterface>(op))
    return op->emitOpError(
        "call-like operation cannot be imported into egraph");

  if (op->getNumRegions() != 0)
    return op->emitOpError(
        "operation with nested regions cannot be imported into egraph");

  if (!isPure(op))
    return op->emitOpError(
        "operation is not pure and cannot be imported into egraph");

  for (OpOperand &operand : op->getOpOperands())
    if (failed(verifyOperandIsBlockLocal(op, operand, block)))
      return failure();

  return verifyResultUsersAreBlockLocal(op, block);
}

LogicalResult verifyFuncIsEligibleForImport(func::FuncOp func) {
  LogicalResult result = success();
  for (Block &block : func.getBody())
    for (Operation &op : block)
      if (failed(verifyOperationIsEligibleForImport(&op, block)))
        result = failure();

  return result;
}

FailureOr<EGraphOp> importFuncToEGraph(func::FuncOp func) {
  ModuleOp module = func->getParentOfType<ModuleOp>();
  if (!module)
    return func.emitOpError("must be nested in a module to import to egraph");
  if (func.isExternal())
    return func.emitOpError("cannot import an external function to egraph");
  if (failed(verifyFuncIsEligibleForImport(func)))
    return failure();
  if (!llvm::hasSingleElement(func.getBody()))
    return func.emitOpError(
        "func-to-egraph skeleton currently requires a single-block function");

  Block &funcEntry = func.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(funcEntry.getTerminator());
  if (!returnOp)
    return func.emitOpError(
        "func-to-egraph skeleton currently requires func.return");

  DeterministicSymbolNamer moduleNamer(module);
  std::string egraphName = moduleNamer.claimEGraphName(func);

  OpBuilder builder(module.getContext());
  builder.setInsertionPointAfter(func);

  SmallVector<NamedAttribute> egraphAttrs;
  if (Attribute visibility =
          func->getAttr(SymbolTable::getVisibilityAttrName())) {
    egraphAttrs.push_back(
        builder.getNamedAttr(SymbolTable::getVisibilityAttrName(), visibility));
  }

  EGraphOp egraph = EGraphOp::create(func.getLoc(), egraphName,
                                     func.getFunctionType(), egraphAttrs);
  builder.insert(egraph.getOperation());
  Block *egraphEntry = egraph.addEntryBlock();

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(egraphEntry);
  DeterministicSymbolNamer egraphNamer(egraph.getOperation());
  DenseMap<Value, FlatSymbolRefAttr> importedSymbols;
  for (auto indexedArg : llvm::enumerate(egraphEntry->getArguments())) {
    unsigned argIndex = indexedArg.index();
    BlockArgument egraphArg = indexedArg.value();
    std::string inputName = egraphNamer.claimInputName(argIndex);
    auto nameAttr = bodyBuilder.getStringAttr(inputName);
    InputOp::create(bodyBuilder, func.getLoc(), nameAttr, egraphArg,
                    TypeAttr::get(egraphArg.getType()));
    // Bind the function argument to the stable input symbol before importing
    // users so operand lookup can resolve through the symbol table.
    importedSymbols.try_emplace(funcEntry.getArgument(argIndex),
                                FlatSymbolRefAttr::get(nameAttr));
  }

  for (Operation &op : funcEntry.without_terminator())
    if (failed(importOperationToEClass(&op, bodyBuilder, egraphNamer,
                                       importedSymbols)))
      return failure();

  SmallVector<Attribute> targets;
  SmallVector<Type> resultTypes;
  for (Value operand : returnOp.getOperands()) {
    auto symbolIt = importedSymbols.find(operand);
    if (symbolIt == importedSymbols.end())
      return returnOp.emitOpError("could not resolve return operand to egraph "
                                  "symbol");

    targets.push_back(symbolIt->second);
    resultTypes.push_back(operand.getType());
  }

  ReturnOp::create(bodyBuilder, returnOp.getLoc(),
                   bodyBuilder.getArrayAttr(targets),
                   bodyBuilder.getTypeArrayAttr(resultTypes));
  return egraph;
}

struct ConvertFuncToEGraphPass
    : public PassWrapper<ConvertFuncToEGraphPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertFuncToEGraphPass)

  StringRef getArgument() const final { return "convert-func-to-egraph"; }

  StringRef getDescription() const final {
    return "convert func.func signatures into symbolic egraph skeletons";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<EGraphDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    for (func::FuncOp func :
         llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      if (failed(importFuncToEGraph(func))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::egraph::createConvertFuncToEGraphPass() {
  return std::make_unique<ConvertFuncToEGraphPass>();
}
