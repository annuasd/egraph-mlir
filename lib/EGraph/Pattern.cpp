#include "MLIREGraph/EGraph/Pattern.h"

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

using namespace mlir;
using namespace mlir::egraph;

#define DEBUG_TYPE "egraph-pattern"

namespace mlir {
namespace egraph {
FailureOr<EGraphMatchStats>
applyEGraphPatterns(EGraph &graph, const EGraphPatternSet &patterns,
                    Operation *rebuildRoot, const EGraphMatchConfig &config);
FailureOr<EGraphMatchStats>
applyEGraphPatterns(EGraph &graph, const EGraphPatternSet &patterns,
                    Operation *rebuildRoot);
} // namespace egraph
} // namespace mlir

namespace {
Type getEventValuePayloadType(Value value) { return value.getType(); }

LogicalResult validateLiveEGraphOwnedOp(EGraph &graph, EOpRefBase ref) {
  if (!ref || ref.getGraph() != &graph || !ref.isLive())
    return failure();
  // Pattern dispatch only accepts live, egraph-owned occurrences.
  if (graph.getOperationOwnership(ref.getOperation()) !=
      OperationOwnership::EGraphOwned)
    return failure();
  return success();
}

bool isSameWorklistRef(EOpRefBase lhs, EOpRefBase rhs) { return lhs == rhs; }

bool containsWorklistRef(ArrayRef<EOpRefBase> refs, EOpRefBase ref) {
  return llvm::any_of(refs, [&](EOpRefBase existing) {
    return isSameWorklistRef(existing, ref);
  });
}

void printDriverDebugValue(raw_ostream &os, EValue value) {
  if (StringAttr symbolName = value.getSymbolNameAttr()) {
    os << '@' << symbolName.getValue();
    if (unsigned resultIndex = value.getResultIndex())
      os << '#' << resultIndex;
    return;
  }

  os << "<value>";
  if (Type type = value.getType())
    os << ':' << type;
}

void printDriverDebugValues(raw_ostream &os, ArrayRef<EValue> values) {
  if (values.size() != 1)
    os << '[';
  llvm::interleaveComma(
      values, os, [&](EValue value) { printDriverDebugValue(os, value); });
  if (values.size() != 1)
    os << ']';
}

void printDriverDebugSymbols(raw_ostream &os, ArrayRef<StringAttr> symbols) {
  os << '[';
  llvm::interleaveComma(
      symbols, os, [&](StringAttr symbol) { os << '@' << symbol.getValue(); });
  os << ']';
}

void printDriverStructuralKey(raw_ostream &os, const EGraphStructuralKey &key) {
  os << "{op=" << key.operationName.getStringRef() << ", child_leader_symbols=";
  printDriverDebugSymbols(os, key.childLeaderSymbols);
  os << '}';
}

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

private:
  Operation *symbolTableOp;
  llvm::StringSet<> reservedNames;
  llvm::StringMap<unsigned> nextSuffixByStem;
};

FailureOr<EClassOp> importOperationToEClass(
    Operation *op, OpBuilder &builder, DeterministicSymbolNamer &namer,
    DenseMap<Value, FlatSymbolRefAttr> &importedSymbols) {
  if (op->getNumResults() != 1)
    return op->emitOpError(
        "block import currently supports only single-result pure operations");

  SmallVector<Attribute> childRefs;
  childRefs.reserve(op->getNumOperands());
  for (OpOperand &operand : op->getOpOperands()) {
    auto symbolIt = importedSymbols.find(operand.get());
    if (symbolIt == importedSymbols.end())
      return op->emitOpError("operand #")
             << operand.getOperandNumber()
             << " has not been imported into the temporary egraph";
    childRefs.push_back(symbolIt->second);
  }

  Value result = op->getResult(0);
  std::string eclassName = namer.claim(getAsmResultName(op, 0).empty()
                                           ? getLeafOperationName(op)
                                           : getAsmResultName(op, 0));
  auto nameAttr = builder.getStringAttr(eclassName);
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
              "are not supported by block import";
  }

  Operation *definingOp = value.getDefiningOp();
  if (definingOp && definingOp->getBlock() == &block)
    return success();

  return op->emitOpError("operand #")
         << operand.getOperandNumber()
         << " uses a value from another block; cross-block values are not "
            "supported by block import";
}

LogicalResult verifyResultUsersAreBlockLocal(Operation *op, Block &block) {
  for (auto indexedResult : llvm::enumerate(op->getResults())) {
    if (!indexedResult.value().isUsedOutsideOfBlock(&block))
      continue;

    return op->emitOpError("result #")
           << indexedResult.index()
           << " is used outside the defining block; cross-block values are not "
              "supported by block import";
  }

  return success();
}

LogicalResult verifyOperationIsEligibleForImport(Operation *op, Block &block) {
  if (op->getNumSuccessors() != 0 || isa<BranchOpInterface>(op) ||
      isa<RegionBranchOpInterface>(op))
    return op->emitOpError(
        "control-flow operation cannot be imported into the temporary egraph");

  if (isa<CallOpInterface>(op))
    return op->emitOpError(
        "call-like operation cannot be imported into the temporary egraph");

  if (op->getNumRegions() != 0)
    return op->emitOpError(
        "operation with nested regions cannot be imported into the temporary "
        "egraph");

  if (!isPure(op))
    return op->emitOpError(
        "operation is not pure and cannot be imported into the temporary "
        "egraph");

  for (OpOperand &operand : op->getOpOperands())
    if (failed(verifyOperandIsBlockLocal(op, operand, block)))
      return failure();

  return verifyResultUsersAreBlockLocal(op, block);
}

LogicalResult verifyBlockIsEligibleForImport(Block &block) {
  if (block.empty())
    return failure();

  Operation *terminator = block.getTerminator();
  if (!terminator)
    return failure();

  LogicalResult result = success();
  for (Operation &op : block.without_terminator())
    if (failed(verifyOperationIsEligibleForImport(&op, block)))
      result = failure();

  for (OpOperand &operand : terminator->getOpOperands())
    if (failed(verifyOperandIsBlockLocal(terminator, operand, block)))
      result = failure();

  return result;
}

FailureOr<OwningOpRef<Operation *>>
importBlockToMatchGraph(Block &block) {
  if (failed(verifyBlockIsEligibleForImport(block)))
    return failure();

  Operation *terminator = block.getTerminator();
  Operation *anchor = block.getParentOp() ? block.getParentOp() : terminator;
  if (!anchor)
    return failure();

  anchor->getContext()->getOrLoadDialect<EGraphDialect>();

  SmallVector<Type, 4> inputTypes;
  for (BlockArgument argument : block.getArguments())
    inputTypes.push_back(argument.getType());
  SmallVector<Type, 4> resultTypes;
  for (Value value : terminator->getOperands())
    resultTypes.push_back(value.getType());

  Builder builder(terminator->getContext());
  FunctionType functionType =
      builder.getFunctionType(inputTypes, resultTypes);
  std::string egraphName = "__match_egraph";

  EGraphOp egraph = EGraphOp::create(anchor->getLoc(), egraphName, functionType);
  OwningOpRef<Operation *> ownedEgraph(egraph.getOperation());
  EGraphOp ownedEgraphOp = cast<EGraphOp>(ownedEgraph.get());

  Block *entryBlock = ownedEgraphOp.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entryBlock);
  DeterministicSymbolNamer namer(ownedEgraph.get());
  DenseMap<Value, FlatSymbolRefAttr> importedSymbols;

  for (auto indexedArg : llvm::enumerate(entryBlock->getArguments())) {
    unsigned argIndex = indexedArg.index();
    BlockArgument egraphArg = indexedArg.value();
    std::string inputName = namer.claim((Twine("arg") + Twine(argIndex)).str());
    auto nameAttr = bodyBuilder.getStringAttr(inputName);
    InputOp::create(bodyBuilder, egraphArg.getLoc(), nameAttr, egraphArg,
                    TypeAttr::get(egraphArg.getType()));
    importedSymbols.try_emplace(block.getArgument(argIndex),
                                FlatSymbolRefAttr::get(nameAttr));
  }

  for (Operation &op : block.without_terminator())
    if (failed(importOperationToEClass(&op, bodyBuilder, namer,
                                       importedSymbols)))
      return failure();

  SmallVector<Attribute> targets;
  for (Value operand : terminator->getOperands()) {
    auto symbolIt = importedSymbols.find(operand);
    if (symbolIt == importedSymbols.end())
      return terminator->emitOpError("could not resolve terminator operand to "
                                     "temporary egraph symbol");
    targets.push_back(symbolIt->second);
  }

  SmallVector<Type> targetTypes(resultTypes.begin(), resultTypes.end());
  ReturnOp::create(bodyBuilder, terminator->getLoc(),
                   bodyBuilder.getArrayAttr(targets),
                   bodyBuilder.getTypeArrayAttr(targetTypes));
  return ownedEgraph;
}

void emitDriverDebug(raw_ostream *stream, StringRef line) {
  if (!stream)
    return;
  *stream << line << '\n';
}

void emitDriverDebugCandidate(raw_ostream *stream, EGraph &graph,
                              StringRef prefix, EOpRefBase ref) {
  if (!stream || !ref || !ref.isLive())
    return;

  *stream << prefix << ": candidate " << ref.getOperationName()
          << " in symbol @" << ref.getEClassOp().getSymName()
          << " with leader symbols ";
  SmallVector<StringAttr, 4> leaderSymbols;
  for (unsigned resultSlot : ref.getYieldedResultSlots()) {
    EValue leader = ref.getResult(resultSlot).getLeader();
    if (leader.getSymbolNameAttr())
      leaderSymbols.push_back(leader.getSymbolNameAttr());
  }
  printDriverDebugSymbols(*stream, leaderSymbols);

  FailureOr<EGraphStructuralKey> key = graph.getStructuralKey(ref);
  if (succeeded(key)) {
    *stream << " and structural key ";
    printDriverStructuralKey(*stream, *key);
  }
  *stream << '\n';
}

bool blockContainsNestedRegionOps(Block &block) {
  return llvm::any_of(block,
                      [](Operation &op) { return op.getNumRegions() != 0; });
}

void accumulateMatchStats(EGraphMatchStats &accumulator,
                          const EGraphMatchStats &stats) {
  accumulator.changed |= stats.changed;
  accumulator.limitReached |= stats.limitReached;
  accumulator.iterations += stats.iterations;
  accumulator.enqueuedCandidates += stats.enqueuedCandidates;
  accumulator.skippedStaleRefs += stats.skippedStaleRefs;
  accumulator.matchedPatterns += stats.matchedPatterns;
  accumulator.changedCommits += stats.changedCommits;
  accumulator.rebuilds += stats.rebuilds;
  if (accumulator.reachedLimit == EGraphMatchLimit::None &&
      stats.reachedLimit != EGraphMatchLimit::None)
    accumulator.reachedLimit = stats.reachedLimit;
}

template <typename CallbackT>
LogicalResult runBlockMatches(Operation *op, bool recurseIntoNestedBlocks,
                              CallbackT &&callback) {
  if (!recurseIntoNestedBlocks) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        if (failed(callback(block)))
          return failure();
    return success();
  }

  bool failedAny = false;
  op->walk([&](Block *block) {
    if (blockContainsNestedRegionOps(*block))
      return WalkResult::advance();
    if (failed(callback(*block)))
      failedAny = true;
    return failedAny ? WalkResult::interrupt() : WalkResult::advance();
  });
  return failure(failedAny);
}

void emitTransactionAddDebug(raw_ostream &os,
                             const EGraphInternedOperation &interned) {
  os << "commit add: ";
  if (interned.operation)
    os << interned.operation->getName();
  else
    os << "<null op>";
  os << " -> ";
  printDriverDebugValues(os, interned.results);
  os << (interned.inserted ? " (new)" : " (reused)");
}

void emitTransactionUnionDebug(raw_ostream &os, ArrayRef<EValue> lhsValues,
                               ArrayRef<EValue> rhsValues, bool changed) {
  os << "commit union: ";
  if (lhsValues.size() == 1) {
    printDriverDebugValue(os, lhsValues.front());
    os << " -> ";
    printDriverDebugValue(os, rhsValues.front());
  } else {
    os << '[';
    for (auto indexedPair :
         llvm::enumerate(llvm::zip_equal(lhsValues, rhsValues))) {
      if (indexedPair.index())
        os << ", ";
      auto [lhsValue, rhsValue] = indexedPair.value();
      printDriverDebugValue(os, lhsValue);
      os << " -> ";
      printDriverDebugValue(os, rhsValue);
    }
    os << ']';
  }
  os << " (changed=" << (changed ? "true" : "false") << ')';
}

Value resolveFoldOperand(EGraph &graph, EValue operand) {
  if (!operand || operand.getGraph() != &graph)
    return Value();

  // Folding only sees constant payloads that are already present in the
  // current egraph snapshot.
  for (EOpRefBase def : operand.getDefs()) {
    if (!matchPattern(def.getOperation(), m_Constant()))
      continue;
    return def.getOperation()->getResult(0);
  }

  return Value();
}

LogicalResult materializeFoldResult(EGraphPatternRewriter &rewriter,
                                    Operation *foldedOp, unsigned resultIndex,
                                    OpFoldResult foldResult,
                                    SmallVectorImpl<Value> &replacementValues) {
  if (auto repl = llvm::dyn_cast_if_present<Value>(foldResult)) {
    replacementValues.push_back(repl);
    return success();
  }

  if (!foldedOp || resultIndex >= foldedOp->getNumResults())
    return failure();

  Attribute attr = cast<Attribute>(foldResult);
  Dialect *dialect = foldedOp->getDialect();
  if (!dialect)
    return failure();

  Operation *constOp = dialect->materializeConstant(
      rewriter, attr, foldedOp->getResult(resultIndex).getType(),
      foldedOp->getLoc());
  if (!constOp || constOp->getNumResults() != 1)
    return failure();

  replacementValues.push_back(constOp->getResult(0));
  return success();
}

LogicalResult tryFoldCandidate(EOpRefBase root,
                               EGraphPatternRewriter &rewriter) {
  Operation *operation = root.getOperation();
  // tryFoldCandidate currently supports only single-result, region-free,
  // successor-free ops.
  if (!operation || operation->getNumResults() != 1 ||
      operation->getNumRegions() != 0 || operation->getNumSuccessors() != 0) {
    return failure();
  }

  EGraph &graph = rewriter.getTransaction().getGraph();
  IRMapping mapper;
  for (auto indexedOperand : llvm::enumerate(operation->getOperands())) {
    Value operand = indexedOperand.value();
    Value foldOperand =
        resolveFoldOperand(graph, root.getOperand(indexedOperand.index()));
    mapper.map(operand, foldOperand ? foldOperand : operand);
  }

  // Clone the candidate into scratch first so fold can run against a local
  // payload DAG without mutating the persistent egraph.
  Operation *foldedOp = rewriter.clone(*operation, mapper);
  SmallVector<OpFoldResult, 4> foldResults;
  if (failed(foldedOp->fold(foldResults))) {
    return failure();
  }

  SmallVector<Value, 4> replacementValues;
  if (foldResults.empty()) {
    replacementValues.append(foldedOp->getResults().begin(),
                             foldedOp->getResults().end());
  } else {
    replacementValues.reserve(foldResults.size());
    for (auto indexedFoldResult : llvm::enumerate(foldResults)) {
      if (failed(materializeFoldResult(
              rewriter, foldedOp, indexedFoldResult.index(),
              indexedFoldResult.value(), replacementValues)))
        return failure();
    }
  }

  if (replacementValues.empty()) {
    return failure();
  }

  return rewriter.replaceOp(root, replacementValues);
}
} // namespace

StringRef mlir::egraph::stringifyEGraphMatchLimit(EGraphMatchLimit limit) {
  switch (limit) {
  case EGraphMatchLimit::None:
    return "none";
  case EGraphMatchLimit::Iteration:
    return "iteration";
  case EGraphMatchLimit::Candidate:
    return "candidate";
  case EGraphMatchLimit::Rebuild:
    return "rebuild";
  }
  llvm_unreachable("unexpected match limit");
}

void mlir::egraph::printEGraphMatchStats(llvm::raw_ostream &os,
                                         const EGraphMatchStats &stats) {
  os << "limit=" << stringifyEGraphMatchLimit(stats.reachedLimit)
     << " limit_reached=" << (stats.limitReached ? "true" : "false")
     << " iterations=" << stats.iterations
     << " enqueued_candidates=" << stats.enqueuedCandidates
     << " skipped_stale_refs=" << stats.skippedStaleRefs
     << " matched_patterns=" << stats.matchedPatterns
     << " changed_commits=" << stats.changedCommits
     << " rebuilds=" << stats.rebuilds
     << " changed=" << (stats.changed ? "true" : "false");
}

GraphMatchState::GraphMatchState(Block &block, std::unique_ptr<EGraph> graph,
                                 OwningOpRef<Operation *> egraphOp,
                                 EGraphMatchStats stats)
    : block(&block), graph(std::move(graph)), egraphOp(std::move(egraphOp)),
      stats(std::move(stats)) {}

Block &GraphMatchState::getBlock() const { return *block; }

const EGraphMatchStats &GraphMatchState::getStats() const { return stats; }

class EGraphRewriteTransaction::ScratchListener final
    : public OpBuilder::Listener {
public:
  explicit ScratchListener(EGraphRewriteTransaction &transaction)
      : transaction(transaction) {}

  void notifyOperationInserted(Operation *operation,
                               OpBuilder::InsertPoint previous) final {
    (void)previous;
    transaction.notifyScratchOperationInserted(operation);
  }

private:
  EGraphRewriteTransaction &transaction;
};

EGraphRewriteTransaction::EGraphRewriteTransaction(EGraph &graph,
                                                   MLIRContext *context)
    : graph(&graph), context(context), scratchBlock(std::make_unique<Block>()),
      listener(std::make_unique<ScratchListener>(*this)) {
  assert(context && "expected an MLIR context for scratch rewriting");
}

EGraphRewriteTransaction::~EGraphRewriteTransaction() { discard(); }

void EGraphRewriteTransaction::discard() {
  if (discarded)
    return;

  // Tear down scratch DAGs in reverse insertion order so defining values do not
  // outlive their users during block destruction.
  for (Operation *operation : llvm::reverse(scratchOperations)) {
    graph->unregisterScratchOperation(operation);
    operation->erase();
  }
  scratchOperations.clear();
  rewriteEvents.clear();
  scratchBlock.reset();
  discarded = true;
}

OpBuilder::Listener *EGraphRewriteTransaction::getListener() const {
  return listener.get();
}

void EGraphRewriteTransaction::notifyScratchOperationInserted(
    Operation *operation) {
  assert(scratchBlock && "cannot insert into a discarded scratch block");
  assert(operation->getBlock() == scratchBlock.get() &&
         "scratch rewriter inserted outside its transaction block");

  scratchOperations.push_back(operation);
  graph->registerScratchOperation(operation);
}

LogicalResult EGraphRewriteTransaction::validateEvents() const {
  if (discarded)
    return failure();

  for (const EGraphRewriteEvent &event : rewriteEvents) {
    if (failed(validateLiveEGraphOwnedOp(*graph, event.root)))
      return failure();

    Operation *oldOperation = event.root.getOperation();
    if (oldOperation->getNumResults() != event.replacementValues.size())
      return failure();

    for (auto indexedValue : llvm::enumerate(event.replacementValues)) {
      Value replacementValue = indexedValue.value();
      Type expectedType =
          oldOperation->getResult(indexedValue.index()).getType();
      if (!replacementValue ||
          getEventValuePayloadType(replacementValue) != expectedType)
        return failure();

      if (succeeded(graph->lookupValue(replacementValue)))
        continue;
      OperationOwnership ownership = graph->getValueOwnership(replacementValue);
      if (ownership == OperationOwnership::ScratchCreated &&
          ownsScratchValue(replacementValue))
        continue;
      if (graph->isLegalAliasValue(replacementValue))
        continue;

      return failure();
    }
  }

  return success();
}

FailureOr<SmallVector<Operation *>>
EGraphRewriteTransaction::collectReachableScratchOperations() const {
  if (failed(validateEvents()))
    return failure();

  llvm::SmallPtrSet<Operation *, 8> visited;
  SmallVector<Operation *> ordered;
  for (const EGraphRewriteEvent &event : rewriteEvents) {
    for (Value replacementValue : event.replacementValues)
      if (failed(
              collectReachableScratchValue(replacementValue, visited, ordered)))
        return failure();
  }

  return ordered;
}

FailureOr<EGraphTransactionInternResult>
EGraphRewriteTransaction::internReachableScratchOperations() {
  FailureOr<SmallVector<Operation *>> reachableOperations =
      collectReachableScratchOperations();
  if (failed(reachableOperations))
    return failure();

  Operation *insertionAnchor = nullptr;
  if (!rewriteEvents.empty())
    insertionAnchor = rewriteEvents.front().root.getEClassOp().getOperation();
  if (!insertionAnchor)
    return failure();

  llvm::DenseMap<Value, EValue> scratchResultValues;
  auto resolveValue = [&](Value value) -> FailureOr<EValue> {
    if (!value)
      return failure();

    auto scratchIt = scratchResultValues.find(value);
    if (scratchIt != scratchResultValues.end())
      return scratchIt->second;

    FailureOr<EValue> lookedUp = graph->lookupValue(value);
    if (succeeded(lookedUp))
      return *lookedUp;

    return failure();
  };

  EGraphTransactionInternResult result;
  for (Operation *operation : *reachableOperations) {
    SmallVector<EValue, 4> operands;
    operands.reserve(operation->getNumOperands());
    for (Value operand : operation->getOperands()) {
      FailureOr<EValue> internedOperand = resolveValue(operand);
      if (failed(internedOperand))
        return failure();
      operands.push_back(*internedOperand);
    }

    FailureOr<EGraphInternedOperation> interned =
        graph->intern(operation, operands, insertionAnchor);
    if (failed(interned))
      return failure();

    if (interned->results.size() != operation->getNumResults())
      return failure();

    for (auto indexedResult : llvm::enumerate(operation->getResults()))
      scratchResultValues[indexedResult.value()] =
          interned->results[indexedResult.index()];

    LDBG_OS([&](raw_ostream &os) { emitTransactionAddDebug(os, *interned); });

    if (interned->inserted)
      insertionAnchor = interned->eclass.getOperation();
    result.operations.push_back(std::move(*interned));
  }

  auto appendEventValue = [&](Value originalValue,
                              EValue internedValue) -> LogicalResult {
    if (!internedValue || internedValue.getGraph() != graph)
      return failure();
    result.eventValues.push_back({originalValue, internedValue});
    return success();
  };

  auto addEventValue = [&](Value value) -> LogicalResult {
    FailureOr<EValue> internedValue = resolveValue(value);
    if (failed(internedValue))
      return failure();

    return appendEventValue(value, *internedValue);
  };

  auto internEventTuple = [&](ValueRange values) -> LogicalResult {
    for (Value value : values)
      if (failed(addEventValue(value)))
        return failure();
    return success();
  };

  for (const EGraphRewriteEvent &event : rewriteEvents) {
    if (failed(internEventTuple(event.replacementValues)))
      return failure();
  }

  return result;
}

FailureOr<EGraphUnionResult> EGraphRewriteTransaction::unionInternedEventValues(
    const EGraphTransactionInternResult &interned) {
  if (failed(validateEvents()))
    return failure();

  SmallVector<EValue, 4> lhsValues;
  SmallVector<EValue, 4> rhsValues;
  unsigned eventValueIndex = 0;

  auto addEventValuePair = [&](EValue lhs) -> LogicalResult {
    if (eventValueIndex >= interned.eventValues.size())
      return failure();

    EValue rhs = interned.eventValues[eventValueIndex++].value;
    if (!lhs || lhs.getGraph() != graph || !rhs || rhs.getGraph() != graph)
      return failure();

    lhsValues.push_back(lhs);
    rhsValues.push_back(rhs);
    return success();
  };

  for (const EGraphRewriteEvent &event : rewriteEvents) {
    Operation *oldOperation = event.root.getOperation();
    for (unsigned i = 0, e = oldOperation->getNumResults(); i < e; ++i) {
      if (!event.root.hasResult(i) ||
          failed(addEventValuePair(event.root.getResult(i))))
        return failure();
    }
  }

  if (eventValueIndex != interned.eventValues.size())
    return failure();

  FailureOr<EGraphUnionResult> unioned =
      graph->unionValueTuples(lhsValues, rhsValues);
  if (failed(unioned))
    return failure();

  LDBG_OS([&](raw_ostream &os) {
    emitTransactionUnionDebug(os, lhsValues, rhsValues, unioned->changed);
  });
  return unioned;
}

FailureOr<EGraphRewriteCommitResult>
EGraphRewriteTransaction::commit(Operation *rebuildRoot, bool rebuildNow) {
  auto failAndDiscard = [&]() -> FailureOr<EGraphRewriteCommitResult> {
    discard();
    return failure();
  };

  if (discarded || (rebuildNow && !rebuildRoot))
    return failAndDiscard();

  EGraphRewriteCommitResult result;
  if (!hasEvents()) {
    discard();
    return result;
  }

  if (failed(validateEvents()))
    return failAndDiscard();

  FailureOr<EGraphTransactionInternResult> interned =
      internReachableScratchOperations();
  if (failed(interned))
    return failAndDiscard();

  FailureOr<EGraphUnionResult> unioned = unionInternedEventValues(*interned);
  if (failed(unioned))
    return failAndDiscard();

  result.changed = unioned->changed;
  if (!result.changed) {
    discard();
    return result;
  }

  if (rebuildNow) {
    FailureOr<EGraphRebuildResult> rebuilt = graph->rebuild(rebuildRoot);
    if (failed(rebuilt))
      return failAndDiscard();

    result.rebuilds = 1;
    result.newCandidateRoots = std::move(rebuilt->newCandidateRoots);
    result.affectedParentCandidates =
        std::move(rebuilt->affectedParentCandidates);
  }

  discard();
  return result;
}

LogicalResult
EGraphRewriteTransaction::recordRewriteEvent(EGraphRewriteEventKind kind,
                                             EOpRefBase root,
                                             ValueRange replacementValues) {
  if (discarded || !root || root.getGraph() != graph || !root.isLive())
    return failure();

  EGraphRewriteEvent event;
  event.kind = kind;
  event.root = root;
  event.replacementValues.append(replacementValues.begin(),
                                 replacementValues.end());
  rewriteEvents.push_back(std::move(event));
  return success();
}

LogicalResult EGraphRewriteTransaction::recordEquivalence(
    EOpRefBase fromOp, Operation *toOperation, EOpRefBase toOp) {
  if (discarded || !fromOp || fromOp.getGraph() != graph || !fromOp.isLive() ||
      !toOperation)
    return failure();

  OperationOwnership toOwnership = graph->getOperationOwnership(toOperation);
  if (toOwnership == OperationOwnership::IllegalExternal)
    return failure();
  if (toOwnership == OperationOwnership::ScratchCreated &&
      !ownsScratchOperation(toOperation))
    return failure();

  if (toOp) {
    if (toOp.getGraph() != graph || !toOp.isLive() ||
        toOp.getOperation() != toOperation)
      return failure();
  } else if (toOwnership == OperationOwnership::EGraphOwned) {
    FailureOr<EOpRefBase> lookedUpToOp = graph->lookupOpRef(toOperation);
    if (failed(lookedUpToOp))
      return failure();
    toOp = *lookedUpToOp;
  }

  SmallVector<Value, 4> replacementValues(toOperation->getResults().begin(),
                                          toOperation->getResults().end());
  return recordRewriteEvent(EGraphRewriteEventKind::ReplaceAllUsesWith, fromOp,
                            replacementValues);
}

LogicalResult EGraphRewriteTransaction::collectReachableScratchOperation(
    Operation *operation, llvm::SmallPtrSetImpl<Operation *> &visited,
    SmallVectorImpl<Operation *> &ordered) const {
  if (!operation ||
      graph->getOperationOwnership(operation) !=
          OperationOwnership::ScratchCreated ||
      !ownsScratchOperation(operation))
    return failure();

  if (!visited.insert(operation).second)
    return success();

  for (Value operand : operation->getOperands())
    if (failed(collectReachableScratchValue(operand, visited, ordered)))
      return failure();

  ordered.push_back(operation);
  return success();
}

LogicalResult EGraphRewriteTransaction::collectReachableScratchValue(
    Value value, llvm::SmallPtrSetImpl<Operation *> &visited,
    SmallVectorImpl<Operation *> &ordered) const {
  if (!value)
    return failure();

  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *definingOp = result.getOwner();
    if (graph->getOperationOwnership(definingOp) ==
        OperationOwnership::ScratchCreated)
      return collectReachableScratchOperation(definingOp, visited, ordered);
  }

  if (succeeded(graph->lookupValue(value)) || graph->isLegalAliasValue(value))
    return success();

  return failure();
}

bool EGraphRewriteTransaction::ownsScratchOperation(
    Operation *operation) const {
  return llvm::is_contained(scratchOperations, operation);
}

bool EGraphRewriteTransaction::ownsScratchValue(Value value) const {
  auto result = dyn_cast<OpResult>(value);
  return result && ownsScratchOperation(result.getOwner());
}

EGraphPatternRewriter::EGraphPatternRewriter(
    EGraphRewriteTransaction &transaction)
    : OpBuilder(transaction.getContext(), transaction.getListener()),
      transaction(&transaction) {
  OpBuilder::setInsertionPointToEnd(transaction.getScratchBlock());
}

LogicalResult EGraphPatternRewriter::replaceOp(EOpRefBase oldOp,
                                               ValueRange replacementValues) {
  return transaction->recordRewriteEvent(EGraphRewriteEventKind::ReplaceOp,
                                         oldOp, replacementValues);
}

LogicalResult EGraphPatternRewriter::replaceOp(Operation *oldOp,
                                               ValueRange replacementValues) {
  FailureOr<EOpRefBase> oldRef = transaction->getGraph().lookupOpRef(oldOp);
  if (failed(oldRef))
    return failure();
  return replaceOp(*oldRef, replacementValues);
}

LogicalResult EGraphPatternRewriter::replaceAllUsesWith(EOpRefBase fromOp,
                                                        EOpRefBase toOp) {
  if (!toOp || toOp.getGraph() != &transaction->getGraph() || !toOp.isLive())
    return failure();
  return transaction->recordEquivalence(fromOp, toOp.getOperation(), toOp);
}

LogicalResult EGraphPatternRewriter::replaceAllUsesWith(Operation *fromOp,
                                                        Operation *toOp) {
  FailureOr<EOpRefBase> fromRef = transaction->getGraph().lookupOpRef(fromOp);
  if (failed(fromRef))
    return failure();
  return transaction->recordEquivalence(*fromRef, toOp);
}

static FailureOr<EGraphMatchStats>
runEGraphMatchDriver(EGraph &graph, const EGraphPatternSet &patterns,
                     Operation *rebuildRoot,
                     const EGraphMatchConfig &config) {
  if (!rebuildRoot)
    return failure();

  EGraphMatchStats result;
  raw_ostream *debugStream = config.debugStream;

  auto returnResult = [&]() -> FailureOr<EGraphMatchStats> {
    if (debugStream) {
      *debugStream << "driver stats: ";
      printEGraphMatchStats(*debugStream, result);
      *debugStream << '\n';
    }
    return result;
  };

  if (!graph.isClean()) {
    if (failed(graph.rebuild(rebuildRoot)))
      return failure();
    ++result.rebuilds;
    emitDriverDebug(debugStream,
                    "driver debug rebuilt dirty graph before dispatch");
    if (config.maxRebuilds != 0 && result.rebuilds >= config.maxRebuilds) {
      result.limitReached = true;
      result.reachedLimit = EGraphMatchLimit::Rebuild;
      emitDriverDebug(debugStream,
                      "driver debug reached rebuild limit before dispatch");
      return returnResult();
    }
  }

  SmallVector<EOpRefBase, 16> worklist;

  auto recordLimit = [&](EGraphMatchLimit limit) {
    result.limitReached = true;
    result.reachedLimit = limit;
    if (debugStream)
      *debugStream << "driver debug reached "
                   << stringifyEGraphMatchLimit(limit) << " limit\n";
  };

  auto enqueueRefs = [&](ArrayRef<EOpRefBase> refs) {
    for (EOpRefBase ref : refs) {
      if (!ref || containsWorklistRef(worklist, ref))
        continue;
      if (config.maxEnqueuedCandidates != 0 &&
          worklist.size() >= config.maxEnqueuedCandidates) {
        recordLimit(EGraphMatchLimit::Candidate);
        return false;
      }
      worklist.push_back(ref);
      result.enqueuedCandidates = worklist.size();
      emitDriverDebugCandidate(debugStream, graph, "driver debug enqueued",
                               ref);
    }
    return true;
  };

  if (!enqueueRefs(graph.getOpRefs()))
    return returnResult();

  // Match all refs in the current round against a clean snapshot. Applying the
  // transactions is deferred to the round boundary so same-round matches do not
  // observe earlier unions.
  struct PendingRewriteCommit {
    EOpRefBase root;
    std::unique_ptr<EGraphRewriteTransaction> transaction;
  };

  std::vector<PendingRewriteCommit> pendingCommits;

  auto finalizePendingCommits = [&]() -> LogicalResult {
    if (pendingCommits.empty())
      return success();

    bool changedInRound = false;
    for (PendingRewriteCommit &pending : pendingCommits) {
      FailureOr<EGraphRewriteCommitResult> committed =
          pending.transaction->commit(rebuildRoot, /*rebuildNow=*/false);
      if (failed(committed))
        return failure();
      if (!committed->changed) {
        emitDriverDebugCandidate(debugStream, graph,
                                 "driver debug kept no-op success for",
                                 pending.root);
        continue;
      }

      result.changed = true;
      changedInRound = true;
      ++result.changedCommits;
      emitDriverDebugCandidate(debugStream, graph,
                               "driver debug committed changed rewrite for",
                               pending.root);
    }
    pendingCommits.clear();

    if (!changedInRound)
      return success();

    FailureOr<EGraphRebuildResult> rebuilt = graph.rebuild(rebuildRoot);
    if (failed(rebuilt))
      return failure();
    ++result.rebuilds;
    emitDriverDebug(debugStream, "driver debug rebuilt after batched commits");

    if (config.maxRebuilds != 0 && result.rebuilds >= config.maxRebuilds) {
      recordLimit(EGraphMatchLimit::Rebuild);
      return success();
    }

    if (!enqueueRefs(rebuilt->newCandidateRoots) ||
        !enqueueRefs(rebuilt->affectedParentCandidates))
      return success();

    return success();
  };

  // Index-based FIFO keeps enqueue order stable while new refs are appended.
  for (unsigned worklistIndex = 0; worklistIndex < worklist.size();
       ++worklistIndex) {
    if (config.maxIterations != 0 &&
        result.iterations >= config.maxIterations) {
      recordLimit(EGraphMatchLimit::Iteration);
      break;
    }

    EOpRefBase root = worklist[worklistIndex];
    ++result.iterations;

    if (!root.isLive()) {
      ++result.skippedStaleRefs;
      emitDriverDebug(debugStream, "driver debug skipped stale candidate ref");
      continue;
    }

    assert(graph.isClean() &&
           "worklist driver must only dispatch patterns on a clean egraph");

    ArrayRef<EGraphPattern *> matchingPatterns =
        patterns.lookup(root.getOperationName());
    bool foldedRoot = false;
    if (config.enableMlirFold) {
      auto transaction = std::make_unique<EGraphRewriteTransaction>(
          graph, rebuildRoot->getContext());
      EGraphPatternRewriter rewriter(*transaction);
      if (succeeded(tryFoldCandidate(root, rewriter)) &&
          transaction->hasEvents()) {
        pendingCommits.push_back({root, std::move(transaction)});
        foldedRoot = true;
      } else {
        transaction->discard();
      }
    }

    if (!foldedRoot) {
      for (EGraphPattern *pattern : matchingPatterns) {
        if (!root.isLive()) {
          ++result.skippedStaleRefs;
          break;
        }

        auto transaction = std::make_unique<EGraphRewriteTransaction>(
            graph, rebuildRoot->getContext());
        EGraphPatternRewriter rewriter(*transaction);
        if (failed(pattern->matchAndRewrite(root, rewriter))) {
          transaction->discard();
          emitDriverDebugCandidate(debugStream, graph,
                                   "driver debug discarded failed pattern for",
                                   root);
          continue;
        }

        ++result.matchedPatterns;
        if (!transaction->hasEvents()) {
          transaction->discard();
          emitDriverDebugCandidate(debugStream, graph,
                                   "driver debug kept no-op success for", root);
          continue;
        }

        pendingCommits.push_back({root, std::move(transaction)});
      }
    }

    if (worklistIndex + 1 == worklist.size()) {
      // Commit and rebuild only at the round boundary so later matches in the
      // same round still observe the same clean snapshot.
      if (failed(finalizePendingCommits()))
        return failure();
      if (result.limitReached)
        return returnResult();
    }
  }

  pendingCommits.clear();
  return returnResult();
}

FailureOr<EGraphMatchStats> mlir::egraph::applyEGraphPatterns(
    EGraph &graph, const EGraphPatternSet &patterns, Operation *rebuildRoot,
    const EGraphMatchConfig &config) {
  return runEGraphMatchDriver(graph, patterns, rebuildRoot, config);
}

FailureOr<EGraphMatchStats> mlir::egraph::applyEGraphPatterns(
    EGraph &graph, const EGraphPatternSet &patterns, Operation *rebuildRoot) {
  EGraphMatchConfig config;
  return runEGraphMatchDriver(graph, patterns, rebuildRoot, config);
}

FailureOr<GraphMatchState> mlir::egraph::applyEGraphPatterns(
    Block &block, const EGraphPatternSet &patterns,
    const EGraphMatchConfig &config) {
  FailureOr<OwningOpRef<Operation *>> imported = importBlockToMatchGraph(block);
  if (failed(imported))
    return failure();

  auto graph = std::make_unique<EGraph>();
  EGraphOp egraph = cast<EGraphOp>(imported->get());
  if (failed(graph->indexEGraph(egraph)))
    return failure();

  FailureOr<EGraphMatchStats> stats =
      runEGraphMatchDriver(*graph, patterns, egraph.getOperation(), config);
  if (failed(stats))
    return failure();

  return GraphMatchState(block, std::move(graph), std::move(*imported),
                         std::move(*stats));
}

FailureOr<EGraphMatchStats> mlir::egraph::applyEGraphPatterns(
    Operation *op, const EGraphPatternSet &patterns,
    const EGraphMatchConfig &config, bool recurseIntoNestedBlocks) {
  assert(op && "operation must not be null");

  EGraphMatchStats aggregatedStats;
  LogicalResult result = runBlockMatches(
      op, recurseIntoNestedBlocks, [&](Block &block) -> LogicalResult {
        FailureOr<GraphMatchState> state =
            applyEGraphPatterns(block, patterns, config);
        if (failed(state))
          return failure();
        accumulateMatchStats(aggregatedStats, state->getStats());
        return success();
      });
  if (failed(result))
    return failure();
  return aggregatedStats;
}

LogicalResult mlir::egraph::applyEGraphPatternsAndExtract(
    Block &block, const EGraphPatternSet &patterns, EGraphExtractMode mode,
    EGraphExtractCostModel costModel, const EGraphMatchConfig &config) {
  FailureOr<GraphMatchState> state =
      applyEGraphPatterns(block, patterns, config);
  if (failed(state))
    return failure();
  return extractEGraph(*state, mode, costModel);
}

LogicalResult mlir::egraph::applyEGraphPatternsAndExtract(
    Operation *op, const EGraphPatternSet &patterns, EGraphExtractMode mode,
    EGraphExtractCostModel costModel, const EGraphMatchConfig &config,
    bool recurseIntoNestedBlocks) {
  assert(op && "operation must not be null");

  LogicalResult result = runBlockMatches(
      op, recurseIntoNestedBlocks, [&](Block &block) -> LogicalResult {
        return applyEGraphPatternsAndExtract(block, patterns, mode, costModel,
                                             config);
      });
  if (failed(result))
    return failure();
  return success();
}
