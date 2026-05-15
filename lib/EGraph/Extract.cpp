#include "ExtractInternal.h"
#include "MLIREGraph/EGraph/Pattern.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::egraph;

namespace {
struct ExtractMaterializationChoice {
  enum class Kind {
    Input,
    Candidate,
    Alias,
  };

  Kind kind = Kind::Input;
  EOpRefBase candidateRoot;
  StringAttr aliasTarget;
};

class ExtractMaterializer {
public:
  ExtractMaterializer(EGraph &graph, EGraphOp egraph,
                      const EGraphExtractInfo &selection, OpBuilder &builder,
                      ArrayRef<Value> inputValues)
      : graph(graph), egraph(egraph), selection(selection), builder(builder),
        inputValues(inputValues) {}

  FailureOr<SmallVector<Value, 4>> run() {
    if (!graph.isClean() || !egraph || egraph.getBody().empty() ||
        selection.roots.size() != egraph.getNumResults())
      return failure();

    if (failed(buildInputValueMap()) || failed(buildSelectionMap()))
      return failure();

    SmallVector<Value, 4> materializedRoots;
    materializedRoots.reserve(selection.roots.size());
    for (EValue root : selection.roots) {
      if (!root || root.getGraph() != &graph || !root.getSymbolNameAttr() ||
          root.getResultIndex() != 0)
        return failure();

      FailureOr<Value> materialized =
          materializeSymbol(root.getSymbolNameAttr());
      if (failed(materialized))
        return failure();
      materializedRoots.push_back(*materialized);
    }

    return materializedRoots;
  }

private:
  LogicalResult buildSelectionMap() {
    if (selection.selectedEClasses.size() !=
            selection.selectedCandidateRoots.size() ||
        selection.selectedAliasEClasses.size() !=
            selection.selectedAliasTargets.size())
      return failure();

    for (auto indexedSelected : llvm::enumerate(selection.selectedEClasses)) {
      EValue selected = indexedSelected.value();
      StringAttr symbol = selected.getSymbolNameAttr();
      if (!symbol)
        return failure();

      const EOpRefBase &candidateRoot =
          selection.selectedCandidateRoots[indexedSelected.index()];
      if (!candidateRoot || candidateRoot.getGraph() != &graph ||
          !candidateRoot.isLive() ||
          candidateRoot.getEClassOp().getSymNameAttr() != symbol)
        return failure();

      auto inserted = selectionMap.try_emplace(
          symbol, ExtractMaterializationChoice{
                      ExtractMaterializationChoice::Kind::Candidate,
                      candidateRoot, StringAttr()});
      if (!inserted.second)
        return failure();
    }

    for (auto indexedAlias : llvm::enumerate(selection.selectedAliasEClasses)) {
      EValue alias = indexedAlias.value();
      StringAttr symbol = alias.getSymbolNameAttr();
      if (!symbol)
        return failure();

      EValue target = selection.selectedAliasTargets[indexedAlias.index()];
      if (!target || target.getGraph() != &graph || !target.getSymbolNameAttr())
        return failure();

      auto inserted = selectionMap.try_emplace(
          symbol, ExtractMaterializationChoice{
                      ExtractMaterializationChoice::Kind::Alias, EOpRefBase(),
                      target.getSymbolNameAttr()});
      if (!inserted.second)
        return failure();
    }

    return success();
  }

  LogicalResult buildInputValueMap() {
    if (inputValues.size() != egraph.getNumArguments())
      return failure();

    Block &entryBlock = egraph.getBody().front();
    unsigned seenInputs = 0;
    for (InputOp input : entryBlock.getOps<InputOp>()) {
      StringAttr symbol = input.getSymNameAttr();
      if (!symbol)
        return failure();

      auto blockArg = dyn_cast<BlockArgument>(input.getValue());
      if (!blockArg || blockArg.getOwner() != &entryBlock)
        return failure();

      unsigned argIndex = blockArg.getArgNumber();
      if (argIndex >= inputValues.size())
        return failure();

      Value materializedInput = inputValues[argIndex];
      if (!materializedInput ||
          materializedInput.getType() != input.getPayloadType())
        return failure();

      auto inserted = inputValueMap.try_emplace(symbol, materializedInput);
      if (!inserted.second)
        return failure();
      ++seenInputs;
    }

    return seenInputs == egraph.getNumArguments() ? success() : failure();
  }

  FailureOr<Value> materializeSymbol(StringAttr symbol) {
    if (!symbol)
      return failure();

    auto cached = materializedValues.find(symbol);
    if (cached != materializedValues.end())
      return cached->second;

    if (llvm::is_contained(activeSymbols, symbol))
      return failure();
    activeSymbols.push_back(symbol);

    auto choice = selectionMap.find(symbol);
    if (choice != selectionMap.end()) {
      FailureOr<Value> materialized = materializeChoice(choice->second);
      activeSymbols.pop_back();
      if (failed(materialized))
        return failure();

      materializedValues.insert({symbol, *materialized});
      return *materialized;
    }

    auto input = inputValueMap.find(symbol);
    if (input == inputValueMap.end()) {
      activeSymbols.pop_back();
      return failure();
    }

    materializedValues.insert({symbol, input->second});
    activeSymbols.pop_back();
    return input->second;
  }

  FailureOr<Value>
  materializeChoice(const ExtractMaterializationChoice &choice) {
    switch (choice.kind) {
    case ExtractMaterializationChoice::Kind::Input:
      return failure();
    case ExtractMaterializationChoice::Kind::Alias: {
      FailureOr<Value> target = materializeSymbol(choice.aliasTarget);
      if (failed(target))
        return failure();
      return *target;
    }
    case ExtractMaterializationChoice::Kind::Candidate:
      break;
    }

    Operation *operation = choice.candidateRoot.getOperation();
    if (!operation || operation->getNumResults() == 0)
      return failure();

    Block *candidateBlock = operation->getBlock();
    if (!candidateBlock || candidateBlock->empty())
      return failure();

    auto yield = dyn_cast_or_null<YieldOp>(candidateBlock->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();

    IRMapping mapping;
    for (auto indexedArg : llvm::enumerate(candidateBlock->getArguments())) {
      EValue child = choice.candidateRoot.getOperand(indexedArg.index());
      FailureOr<Value> childValue =
          materializeSymbol(child.getSymbolNameAttr());
      if (failed(childValue))
        return failure();

      mapping.map(indexedArg.value(), *childValue);
    }

    for (Operation &candidateOp : *candidateBlock) {
      if (isa<YieldOp>(candidateOp))
        continue;
      if (!builder.clone(candidateOp, mapping))
        return failure();
    }

    Value yieldedValue = mapping.lookupOrDefault(yield.getOperand(0));
    if (!yieldedValue)
      return failure();
    return yieldedValue;
  }

  EGraph &graph;
  EGraphOp egraph;
  const EGraphExtractInfo &selection;
  OpBuilder &builder;
  ArrayRef<Value> inputValues;
  DenseMap<StringAttr, ExtractMaterializationChoice> selectionMap;
  DenseMap<StringAttr, Value> inputValueMap;
  DenseMap<StringAttr, Value> materializedValues;
  SmallVector<StringAttr, 4> activeSymbols;
};

FailureOr<SmallVector<Value, 4>>
materializeExtractSelection(EGraph &graph, EGraphOp egraph,
                            const EGraphExtractInfo &selection,
                            OpBuilder &builder, ArrayRef<Value> inputValues) {
  ExtractMaterializer materializer(graph, egraph, selection, builder,
                                   inputValues);
  return materializer.run();
}

LogicalResult
appendNormalizedExtractRoot(EGraph &graph, EGraphOp egraph, unsigned rootIndex,
                            EValue root, Type expectedType,
                            SmallVectorImpl<EValue> &normalizedRoots) {
  if (!root || root.getGraph() != &graph || !root.getSymbolNameAttr())
    return egraph.emitOpError("extract root #")
           << rootIndex
           << " must be a symbol-backed EValue from the current graph";

  if (root.getResultIndex() != 0)
    return egraph.emitOpError("extract root #")
           << rootIndex << " must use result slot 0";

  EValue leader = root.getLeader();
  if (!leader.getSymbolNameAttr() || leader.getResultIndex() != 0)
    return egraph.emitOpError("extract root #")
           << rootIndex << " must resolve to a single-result leader symbol";

  Type actualType = leader.getType();
  if (!actualType)
    return egraph.emitOpError("extract root #")
           << rootIndex << " must reference a payload-bearing egraph symbol";

  if (actualType != expectedType)
    return egraph.emitOpError("extract root #")
           << rootIndex << " type " << actualType
           << " must match enclosing egraph result type " << expectedType;

  normalizedRoots.push_back(leader);
  return success();
}

FailureOr<mlir::egraph::detail::EGraphExtractRequest>
buildRequestFromRoots(EGraph &graph, EGraphOp egraph,
                      ArrayRef<EValue> rootsToNormalize,
                      EGraphExtractMode mode) {
  ArrayRef<Type> resultTypes = egraph.getResultTypes();
  if (rootsToNormalize.size() != resultTypes.size())
    return failure();

  mlir::egraph::detail::EGraphExtractRequest request;
  request.mode = mode;
  request.roots.reserve(rootsToNormalize.size());
  for (auto indexedRoot : llvm::enumerate(rootsToNormalize)) {
    if (failed(appendNormalizedExtractRoot(
            graph, egraph, indexedRoot.index(), indexedRoot.value(),
            resultTypes[indexedRoot.index()], request.roots)))
      return failure();
  }
  return request;
}

FailureOr<mlir::egraph::detail::EGraphExtractRequest>
buildExtractRequest(EGraph &graph, EGraphOp egraph,
                    ArrayRef<EValue> explicitRoots, EGraphExtractMode mode) {
  if (!egraph || egraph.isExternal() || egraph.getBody().empty())
    return failure();

  if (!graph.isClean())
    return failure();

  if (!explicitRoots.empty())
    return buildRequestFromRoots(graph, egraph, explicitRoots, mode);

  auto returnOp =
      dyn_cast_or_null<ReturnOp>(egraph.getBody().front().getTerminator());
  if (!returnOp)
    return failure();

  ArrayAttr targets = returnOp.getTargets();
  if (targets.size() != egraph.getNumResults())
    return failure();

  SmallVector<EValue, 4> returnRoots;
  returnRoots.reserve(targets.size());
  for (Attribute attr : targets) {
    auto target = cast<FlatSymbolRefAttr>(attr);
    returnRoots.push_back(graph.getValue(target));
  }

  return buildRequestFromRoots(graph, egraph, returnRoots, mode);
}

FailureOr<EGraphExtractInfo>
runExtract(EGraph &graph, EGraphOp egraph,
           const mlir::egraph::detail::EGraphExtractRequest &request,
           EGraphExtractCostModel costModel) {
  switch (request.mode) {
  case EGraphExtractMode::Greedy:
    return mlir::egraph::detail::runGreedyExtract(graph, egraph, request,
                                                  costModel);
  case EGraphExtractMode::Z3LinearProgramming:
#ifdef MLIR_EGRAPH_ENABLE_Z3
    return mlir::egraph::detail::runZ3LinearProgrammingExtract(
        graph, egraph, request, costModel);
#else
    return failure();
#endif
  case EGraphExtractMode::OrToolsLinearProgramming:
#ifdef MLIR_EGRAPH_ENABLE_OR_TOOLS
    return mlir::egraph::detail::runOrToolsLinearProgrammingExtract(
        graph, egraph, request, costModel);
#else
    return failure();
#endif
  }
  llvm_unreachable("unexpected egraph extract mode");
}
} // namespace

StringRef mlir::egraph::stringifyEGraphExtractMode(EGraphExtractMode mode) {
  switch (mode) {
  case EGraphExtractMode::Greedy:
    return "greedy";
  case EGraphExtractMode::Z3LinearProgramming:
    return "lp";
  case EGraphExtractMode::OrToolsLinearProgramming:
    return "or-tools-lp";
  }
  llvm_unreachable("unexpected egraph extract mode");
}

LogicalResult mlir::egraph::extractEGraph(EGraph &graph, EGraphOp egraph,
                                          EGraphExtractMode mode,
                                          EGraphExtractCostModel costModel,
                                          EGraphExtractInfo *info,
                                          ArrayRef<EValue> explicitRoots) {
  FailureOr<mlir::egraph::detail::EGraphExtractRequest> request =
      buildExtractRequest(graph, egraph, explicitRoots, mode);
  if (failed(request))
    return failure();

  FailureOr<EGraphExtractInfo> selection =
      runExtract(graph, egraph, *request, costModel);
  if (failed(selection))
    return failure();

  if (info)
    *info = *selection;
  return success();
}

FailureOr<SmallVector<Value, 4>> mlir::egraph::materializeEGraphExtractInfo(
    EGraph &graph, EGraphOp egraph, const EGraphExtractInfo &selection,
    OpBuilder &builder, ArrayRef<Value> inputValues) {
  return materializeExtractSelection(graph, egraph, selection, builder,
                                     inputValues);
}

LogicalResult mlir::egraph::extractEGraph(GraphMatchState &state,
                                          EGraphExtractMode mode,
                                          EGraphExtractCostModel costModel,
                                          ArrayRef<EValue> explicitRoots,
                                          EGraphExtractInfo *info) {
  if (state.extracted || !state.block || !state.graph || !state.egraphOp)
    return failure();

  Block &block = state.getBlock();
  Operation *terminator = block.getTerminator();
  if (!terminator)
    return failure();

  EGraphOp egraph = cast<EGraphOp>(state.egraphOp.get());
  FailureOr<mlir::egraph::detail::EGraphExtractRequest> request =
      buildExtractRequest(*state.graph, egraph, explicitRoots, mode);
  if (failed(request))
    return failure();

  FailureOr<EGraphExtractInfo> selection =
      runExtract(*state.graph, egraph, *request, costModel);
  if (failed(selection))
    return failure();

  if (info)
    *info = *selection;

  SmallVector<Operation *, 4> oldOps;
  oldOps.reserve(block.getOperations().size());
  for (Operation &op : block.without_terminator())
    oldOps.push_back(&op);

  Operation *oldTail = oldOps.empty() ? nullptr : oldOps.back();
  OpBuilder builder(terminator->getContext());
  builder.setInsertionPoint(terminator);

  SmallVector<Value, 4> inputValues(block.getArguments().begin(),
                                    block.getArguments().end());
  FailureOr<SmallVector<Value, 4>> roots = materializeExtractSelection(
      *state.graph, egraph, *selection, builder, inputValues);
  if (failed(roots)) {
    for (Operation *op = terminator->getPrevNode(); op && op != oldTail;) {
      Operation *previous = op->getPrevNode();
      op->erase();
      op = previous;
    }
    return failure();
  }

  if (roots->size() != terminator->getNumOperands()) {
    for (Operation *op = terminator->getPrevNode(); op && op != oldTail;) {
      Operation *previous = op->getPrevNode();
      op->erase();
      op = previous;
    }
    return failure();
  }

  terminator->setOperands(*roots);
  for (Operation *op : llvm::reverse(oldOps))
    op->erase();

  state.extracted = true;
  return success();
}
