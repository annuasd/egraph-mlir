#include "MLIREGraph/EGraph/Extract.h"
#include "MLIREGraph/EGraph/Pattern.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <limits>
#include <string>
#include <vector>

#ifdef MLIR_EGRAPH_ENABLE_Z3
#include <z3++.h>
#endif

using namespace mlir;
using namespace mlir::egraph;

namespace mlir {
namespace egraph {
LogicalResult extractEGraphForTesting(EGraph &graph, EGraphOp egraph,
                                      EGraphExtractMode mode,
                                      EGraphExtractCostModel costModel,
                                      EGraphExtractInfo *info,
                                      ArrayRef<EValue> explicitRoots);
FailureOr<SmallVector<Value, 4>>
materializeEGraphExtractInfoForTesting(EGraph &graph, EGraphOp egraph,
                                       const EGraphExtractInfo &selection,
                                       OpBuilder &builder,
                                       ArrayRef<Value> inputValues);
} // namespace egraph
} // namespace mlir

namespace {
FailureOr<SmallVector<FlatSymbolRefAttr, 4>>
readCandidateSymbolRefs(EClassOp eclass, unsigned candidateOrdinal) {
  ArrayAttr candidateRefs = eclass.getCandidateRefs();
  if (candidateOrdinal >= candidateRefs.size())
    return failure();

  auto row = dyn_cast<ArrayAttr>(candidateRefs[candidateOrdinal]);
  if (!row)
    return failure();

  SmallVector<FlatSymbolRefAttr, 4> refs;
  refs.reserve(row.size());
  for (Attribute attr : row) {
    auto symbolRef = dyn_cast<FlatSymbolRefAttr>(attr);
    if (!symbolRef)
      return failure();
    refs.push_back(symbolRef);
  }

  return refs;
}

FailureOr<StringAttr> getAliasCandidateTarget(EClassOp eclass,
                                              unsigned candidateOrdinal) {
  FailureOr<SmallVector<FlatSymbolRefAttr, 4>> refs =
      readCandidateSymbolRefs(eclass, candidateOrdinal);
  if (failed(refs) || refs->size() != 1)
    return failure();

  auto candidateIt = eclass.getCandidates().begin();
  for (unsigned i = 0; i < candidateOrdinal; ++i)
    ++candidateIt;
  Region &candidate = *candidateIt;
  if (candidate.empty() || !llvm::hasSingleElement(candidate))
    return failure();

  Block &block = candidate.front();
  if (block.getNumArguments() != 1)
    return failure();

  auto yield = dyn_cast_or_null<YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != block.getArgument(0))
    return failure();

  return refs->front().getAttr();
}

bool isAliasCandidate(EClassOp eclass, unsigned candidateOrdinal) {
  return succeeded(getAliasCandidateTarget(eclass, candidateOrdinal));
}

struct EGraphExtractRequest {
  EGraphExtractMode mode = EGraphExtractMode::Greedy;
  SmallVector<EValue, 4> roots;
};

struct GreedyExtractSelection {
  enum class Kind {
    Input,
    Candidate,
    Alias,
  };

  EValue eclass;
  EOpRefBase candidateRoot;
  EGraphExtractCost cost = 0;
  EGraphExtractCost subtreeCost = 0;
  EValue aliasTarget;
  Kind kind = Kind::Input;
};

FailureOr<EGraphExtractCost> addExtractCosts(EGraphExtractCost lhs,
                                             EGraphExtractCost rhs) {
  if (rhs > std::numeric_limits<EGraphExtractCost>::max() - lhs)
    return failure();
  return lhs + rhs;
}

FailureOr<std::pair<EOpRefBase, EGraphExtractCost>>
selectBestCandidateForRoot(EGraph &graph, EValue root,
                           EGraphExtractCostModel costModel) {
  if (!root || root.getGraph() != &graph || !root.getSymbolNameAttr())
    return failure();

  EValue leader = root.getLeader();
  if (!leader.getSymbolNameAttr() || leader.getResultIndex() != 0)
    return failure();

  SmallVector<EOpRefBase, 4> candidates = graph.getCandidateRoots(leader);
  if (candidates.empty())
    return failure();

  bool hasBestCandidate = false;
  EOpRefBase bestCandidate;
  EGraphExtractCost bestCost = 0;
  for (EOpRefBase candidate : candidates) {
    Operation *operation = candidate.getOperation();
    if (!operation)
      continue;

    FailureOr<EGraphExtractCost> candidateCost = costModel(operation);
    if (failed(candidateCost))
      continue;

    if (!hasBestCandidate || *candidateCost < bestCost) {
      hasBestCandidate = true;
      bestCandidate = candidate;
      bestCost = *candidateCost;
    }
  }

  if (!hasBestCandidate)
    return failure();
  return std::make_pair(bestCandidate, bestCost);
}

class GreedyEGraphExtractor {
public:
  GreedyEGraphExtractor(EGraph &graph, EGraphOp egraph,
                        EGraphExtractCostModel costModel)
      : graph(graph), egraph(egraph), costModel(costModel) {}

  FailureOr<EGraphExtractInfo> run(const EGraphExtractRequest &request) {
    if (!graph.isClean() || !egraph ||
        request.mode != EGraphExtractMode::Greedy)
      return failure();

    EGraphExtractInfo result;
    result.mode = EGraphExtractMode::Greedy;
    result.roots = request.roots;
    result.rootCosts.reserve(request.roots.size());

    SmallVector<GreedyExtractSelection, 4> rootSelections;
    rootSelections.reserve(request.roots.size());

    for (EValue root : request.roots) {
      FailureOr<GreedyExtractSelection> selection = computeBest(root);
      if (failed(selection))
        return failure();
      rootSelections.push_back(*selection);
    }

    for (const GreedyExtractSelection &selection : rootSelections)
      result.rootCosts.push_back(selection.subtreeCost);

    for (const GreedyExtractSelection &selection : rootSelections)
      if (failed(collectSelected(selection.eclass, result)))
        return failure();

    result.changed = !result.selectedCandidateRoots.empty() ||
                     !result.selectedAliasEClasses.empty();
    return result;
  }

private:
  FailureOr<GreedyExtractSelection> computeBest(EValue root) {
    if (!root || root.getGraph() != &graph || !root.getSymbolNameAttr())
      return failure();

    EValue leader = root.getLeader();
    if (!leader.getSymbolNameAttr() || leader.getResultIndex() != 0 ||
        !leader.getType())
      return failure();

    StringAttr symbol = leader.getSymbolNameAttr();
    auto cached = bestBySymbol.find(symbol);
    if (cached != bestBySymbol.end())
      return cached->second;

    if (llvm::is_contained(activeSymbols, symbol))
      return failure();

    activeSymbols.push_back(symbol);
    FailureOr<GreedyExtractSelection> selection = computeBestUncached(leader);
    activeSymbols.pop_back();
    if (failed(selection))
      return failure();

    bestBySymbol.insert({symbol, *selection});
    return *selection;
  }

  FailureOr<GreedyExtractSelection> computeBestPreservingSymbol(EValue root) {
    if (!root || root.getGraph() != &graph || !root.getSymbolNameAttr())
      return failure();

    StringAttr symbol = root.getSymbolNameAttr();
    auto cached = rawBestBySymbol.find(symbol);
    if (cached != rawBestBySymbol.end())
      return cached->second;

    if (llvm::is_contained(activeRawSymbols, symbol))
      return failure();

    activeRawSymbols.push_back(symbol);
    FailureOr<GreedyExtractSelection> selection =
        computeBestUncachedPreservingSymbol(root);
    activeRawSymbols.pop_back();
    if (failed(selection))
      return failure();

    rawBestBySymbol.insert({symbol, *selection});
    return *selection;
  }

  FailureOr<GreedyExtractSelection> computeBestUncached(EValue leader) {
    bool hasBestCandidate = false;
    GreedyExtractSelection bestSelection;

    if (auto input = egraph.lookupSymbol<InputOp>(leader.getSymbolName())) {
      (void)input;
      return computeInputLeaf(leader);
    }

    auto eclass = egraph.lookupSymbol<EClassOp>(leader.getSymbolName());
    if (!eclass)
      return failure();

    SmallVector<EOpRefBase, 4> candidates = graph.getCandidateRoots(eclass);
    for (EOpRefBase candidate : candidates) {
      FailureOr<GreedyExtractSelection> candidateSelection =
          computeCandidateSelection(candidate);
      if (failed(candidateSelection))
        continue;

      if (!hasBestCandidate ||
          candidateSelection->subtreeCost < bestSelection.subtreeCost) {
        hasBestCandidate = true;
        bestSelection = *candidateSelection;
      }
    }

    for (auto indexedRegion : llvm::enumerate(eclass.getCandidates())) {
      unsigned candidateOrdinal = indexedRegion.index();
      if (!isAliasCandidate(eclass, candidateOrdinal))
        continue;

      FailureOr<GreedyExtractSelection> aliasSelection =
          computeAliasSelection(eclass, candidateOrdinal);
      if (failed(aliasSelection))
        continue;

      if (!hasBestCandidate ||
          aliasSelection->subtreeCost < bestSelection.subtreeCost) {
        hasBestCandidate = true;
        bestSelection = *aliasSelection;
      }
    }

    if (!hasBestCandidate)
      return failure();
    return bestSelection;
  }

  FailureOr<GreedyExtractSelection>
  computeBestUncachedPreservingSymbol(EValue root) {
    bool hasBestCandidate = false;
    GreedyExtractSelection bestSelection;

    if (auto input = egraph.lookupSymbol<InputOp>(root.getSymbolName())) {
      (void)input;
      return computeInputLeaf(root);
    }

    auto eclass = egraph.lookupSymbol<EClassOp>(root.getSymbolName());
    if (!eclass)
      return failure();

    SmallVector<EOpRefBase, 4> candidates = graph.getCandidateRoots(eclass);
    for (EOpRefBase candidate : candidates) {
      FailureOr<GreedyExtractSelection> candidateSelection =
          computeCandidateSelection(candidate);
      if (failed(candidateSelection))
        continue;

      candidateSelection->eclass = root;
      if (!hasBestCandidate ||
          candidateSelection->subtreeCost < bestSelection.subtreeCost) {
        hasBestCandidate = true;
        bestSelection = *candidateSelection;
      }
    }

    for (auto indexedRegion : llvm::enumerate(eclass.getCandidates())) {
      unsigned candidateOrdinal = indexedRegion.index();
      if (!isAliasCandidate(eclass, candidateOrdinal))
        continue;

      FailureOr<GreedyExtractSelection> aliasSelection =
          computeAliasSelection(eclass, candidateOrdinal);
      if (failed(aliasSelection))
        continue;

      aliasSelection->eclass = root;
      if (!hasBestCandidate ||
          aliasSelection->subtreeCost < bestSelection.subtreeCost) {
        hasBestCandidate = true;
        bestSelection = *aliasSelection;
      }
    }

    if (!hasBestCandidate)
      return failure();

    bestSelection.eclass = root;
    return bestSelection;
  }

  FailureOr<GreedyExtractSelection>
  computeAliasSelection(EClassOp eclass, unsigned candidateOrdinal) {
    FailureOr<StringAttr> aliasTarget =
        getAliasCandidateTarget(eclass, candidateOrdinal);
    if (failed(aliasTarget))
      return failure();

    FailureOr<GreedyExtractSelection> child =
        computeBestPreservingSymbol(graph.getValue(*aliasTarget));
    if (failed(child))
      return failure();

    GreedyExtractSelection selection;
    selection.eclass = graph.getValue(eclass.getSymNameAttr());
    selection.aliasTarget = child->eclass;
    selection.kind = GreedyExtractSelection::Kind::Alias;
    selection.subtreeCost = child->subtreeCost;
    return selection;
  }

  FailureOr<GreedyExtractSelection> computeInputLeaf(EValue leader) {
    if (!egraph.lookupSymbol<InputOp>(leader.getSymbolName()))
      return failure();

    GreedyExtractSelection selection;
    selection.eclass = leader;
    selection.kind = GreedyExtractSelection::Kind::Input;
    return selection;
  }

  FailureOr<GreedyExtractSelection>
  computeCandidateSelection(EOpRefBase candidate) {
    Operation *operation = candidate.getOperation();
    if (!operation)
      return failure();

    FailureOr<EGraphExtractCost> localCost = costModel(operation);
    if (failed(localCost))
      return failure();

    FailureOr<EGraphStructuralKey> key = graph.getStructuralKey(candidate);
    if (failed(key))
      return failure();

    EGraphExtractCost totalCost = *localCost;
    for (StringAttr childSymbol : key->childLeaderSymbols) {
      if (!childSymbol)
        return failure();

      FailureOr<GreedyExtractSelection> child =
          computeBest(graph.getValue(childSymbol));
      if (failed(child))
        return failure();

      FailureOr<EGraphExtractCost> updatedCost =
          addExtractCosts(totalCost, child->subtreeCost);
      if (failed(updatedCost))
        return failure();
      totalCost = *updatedCost;
    }

    GreedyExtractSelection selection;
    selection.eclass = candidate.getResult(0).getLeader();
    selection.candidateRoot = candidate;
    selection.cost = *localCost;
    selection.subtreeCost = totalCost;
    selection.kind = GreedyExtractSelection::Kind::Candidate;
    return selection;
  }

  LogicalResult collectSelected(EValue root, EGraphExtractInfo &result,
                                bool preserveSymbol = false) {
    FailureOr<GreedyExtractSelection> selection =
        preserveSymbol ? computeBestPreservingSymbol(root) : computeBest(root);
    if (failed(selection))
      return failure();

    if (selection->kind == GreedyExtractSelection::Kind::Input)
      return success();

    StringAttr symbol = selection->eclass.getSymbolNameAttr();
    if (selection->kind == GreedyExtractSelection::Kind::Alias) {
      if (llvm::is_contained(emittedAliasSymbols, symbol))
        return success();
      if (llvm::is_contained(collectingSymbols, symbol))
        return failure();

      collectingSymbols.push_back(symbol);
      if (failed(collectSelected(selection->aliasTarget, result, true))) {
        collectingSymbols.pop_back();
        return failure();
      }
      collectingSymbols.pop_back();

      emittedAliasSymbols.push_back(symbol);
      result.selectedAliasEClasses.push_back(selection->eclass);
      result.selectedAliasTargets.push_back(selection->aliasTarget);
      return success();
    }

    if (llvm::is_contained(emittedSymbols, symbol))
      return success();
    if (llvm::is_contained(collectingSymbols, symbol))
      return failure();

    collectingSymbols.push_back(symbol);
    FailureOr<EGraphStructuralKey> key =
        graph.getStructuralKey(selection->candidateRoot);
    if (failed(key)) {
      collectingSymbols.pop_back();
      return failure();
    }

    for (StringAttr childSymbol : key->childLeaderSymbols) {
      if (failed(collectSelected(graph.getValue(childSymbol), result))) {
        collectingSymbols.pop_back();
        return failure();
      }
    }
    collectingSymbols.pop_back();

    emittedSymbols.push_back(symbol);
    result.selectedEClasses.push_back(selection->eclass);
    result.selectedCandidateRoots.push_back(selection->candidateRoot);
    result.selectedCandidateCosts.push_back(selection->cost);
    result.selectedSubtreeCosts.push_back(selection->subtreeCost);
    return success();
  }

  EGraph &graph;
  EGraphOp egraph;
  EGraphExtractCostModel costModel;
  llvm::DenseMap<StringAttr, GreedyExtractSelection> bestBySymbol;
  llvm::DenseMap<StringAttr, GreedyExtractSelection> rawBestBySymbol;
  SmallVector<StringAttr, 4> activeSymbols;
  SmallVector<StringAttr, 4> activeRawSymbols;
  SmallVector<StringAttr, 4> collectingSymbols;
  SmallVector<StringAttr, 4> emittedSymbols;
  SmallVector<StringAttr, 4> emittedAliasSymbols;
};

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

#ifdef MLIR_EGRAPH_ENABLE_Z3
FailureOr<EOpRefBase> getCandidateRootRef(EGraph &graph, EClassOp eclass,
                                          unsigned candidateOrdinal) {
  auto candidateIt = eclass.getCandidates().begin();
  for (unsigned i = 0; i < candidateOrdinal; ++i)
    ++candidateIt;
  Region &candidate = *candidateIt;
  if (candidate.empty() || !llvm::hasSingleElement(candidate))
    return failure();

  Block &block = candidate.front();
  auto yield = dyn_cast_or_null<YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return failure();

  Operation *root = yield.getOperand(0).getDefiningOp();
  if (!root || root->getBlock() != &block)
    return failure();

  return graph.lookupOpRef(root);
}

class LinearProgrammingEGraphExtractor {
public:
  LinearProgrammingEGraphExtractor(EGraph &graph, EGraphOp egraph,
                                   EGraphExtractCostModel costModel)
      : graph(graph), egraph(egraph), costModel(costModel) {}

  FailureOr<EGraphExtractInfo> run(const EGraphExtractRequest &request) {
    if (!graph.isClean() || !egraph ||
        request.mode != EGraphExtractMode::LinearProgramming)
      return failure();

    for (EValue root : request.roots) {
      if (!root || root.getGraph() != &graph || !root.getSymbolNameAttr() ||
          root.getResultIndex() != 0)
        return failure();

      FailureOr<unsigned> rootIndex = getOrCreateNode(root.getSymbolNameAttr());
      if (failed(rootIndex))
        return failure();
      if (failed(populateNode(*rootIndex)))
        return failure();
      rootNodeIndices.push_back(*rootIndex);
    }

    if (failed(solve()))
      return failure();

    EGraphExtractInfo result;
    result.mode = EGraphExtractMode::LinearProgramming;
    result.roots = request.roots;
    result.rootCosts.reserve(request.roots.size());

    DenseMap<StringAttr, EGraphExtractCost> subtreeCosts;
    SmallVector<StringAttr, 4> collectingSymbols;
    for (EValue root : request.roots) {
      FailureOr<EGraphExtractCost> rootCost = collectSelected(
          root.getSymbolNameAttr(), result, subtreeCosts, collectingSymbols);
      if (failed(rootCost))
        return failure();
      result.rootCosts.push_back(*rootCost);
    }

    result.changed = !result.selectedCandidateRoots.empty() ||
                     !result.selectedAliasEClasses.empty();
    return result;
  }

private:
  enum class LpChoiceKind {
    Candidate,
    Alias,
  };

  struct LpChoice {
    LpChoiceKind kind = LpChoiceKind::Candidate;
    EOpRefBase candidateRoot;
    EGraphExtractCost cost = 0;
    SmallVector<StringAttr, 4> childSymbols;
    StringAttr aliasTarget;
    unsigned boolVarIndex = ~0u;
  };

  struct LpNode {
    StringAttr symbol;
    bool isInput = false;
    bool populated = false;
    bool populating = false;
    unsigned activeVarIndex = ~0u;
    unsigned rankVarIndex = ~0u;
    SmallVector<LpChoice, 4> choices;
  };

  FailureOr<unsigned> getOrCreateNode(StringAttr symbol) {
    if (!symbol)
      return failure();

    auto existing = nodeIndices.find(symbol);
    if (existing != nodeIndices.end())
      return existing->second;

    auto input = egraph.lookupSymbol<InputOp>(symbol.getValue());
    bool isInput = static_cast<bool>(input);
    if (!isInput && !egraph.lookupSymbol<EClassOp>(symbol.getValue()))
      return failure();

    unsigned index = nodes.size();
    nodeIndices.insert({symbol, index});
    LpNode node;
    node.symbol = symbol;
    node.isInput = isInput;
    nodes.push_back(std::move(node));
    return index;
  }

  LogicalResult populateNode(unsigned nodeIndex) {
    if (nodeIndex >= nodes.size())
      return failure();
    if (nodes[nodeIndex].populated || nodes[nodeIndex].populating)
      return success();

    nodes[nodeIndex].populating = true;
    StringAttr symbol = nodes[nodeIndex].symbol;
    bool isInput = nodes[nodeIndex].isInput;
    if (isInput) {
      nodes[nodeIndex].populating = false;
      nodes[nodeIndex].populated = true;
      return success();
    }

    auto eclass = egraph.lookupSymbol<EClassOp>(symbol.getValue());
    if (!eclass)
      return failure();

    for (auto indexedRegion : llvm::enumerate(eclass.getCandidates())) {
      unsigned candidateOrdinal = indexedRegion.index();
      LpChoice choice;
      if (isAliasCandidate(eclass, candidateOrdinal)) {
        FailureOr<StringAttr> aliasTarget =
            getAliasCandidateTarget(eclass, candidateOrdinal);
        if (failed(aliasTarget))
          continue;

        choice.kind = LpChoiceKind::Alias;
        choice.aliasTarget = *aliasTarget;
        choice.childSymbols.push_back(*aliasTarget);
      } else {
        FailureOr<EOpRefBase> candidateRoot =
            getCandidateRootRef(graph, eclass, candidateOrdinal);
        if (failed(candidateRoot))
          continue;

        Operation *operation = candidateRoot->getOperation();
        if (!operation)
          continue;

        FailureOr<EGraphExtractCost> localCost = costModel(operation);
        if (failed(localCost))
          continue;

        FailureOr<EGraphStructuralKey> key =
            graph.getStructuralKey(*candidateRoot);
        if (failed(key))
          continue;

        choice.kind = LpChoiceKind::Candidate;
        choice.candidateRoot = *candidateRoot;
        choice.cost = *localCost;
        choice.childSymbols.append(key->childLeaderSymbols.begin(),
                                   key->childLeaderSymbols.end());
      }

      unsigned choiceIndex = nodes[nodeIndex].choices.size();
      nodes[nodeIndex].choices.push_back(std::move(choice));
      SmallVector<StringAttr, 4> childSymbols =
          nodes[nodeIndex].choices[choiceIndex].childSymbols;
      for (StringAttr childSymbol : childSymbols) {
        FailureOr<unsigned> childIndex = getOrCreateNode(childSymbol);
        if (failed(childIndex) || failed(populateNode(*childIndex)))
          return failure();
      }
    }

    nodes[nodeIndex].populating = false;
    nodes[nodeIndex].populated = true;
    return success();
  }

  LogicalResult solve() {
    z3::context context;
    z3::optimize optimizer(context);
    z3::params optimizerParams(context);
    optimizerParams.set("priority", "lex");
    optimizer.set(optimizerParams);

    std::vector<z3::expr> boolVars;
    std::vector<z3::expr> rankVars;

    auto addBoolVar = [&](StringRef prefix, StringAttr symbol,
                          unsigned ordinal = ~0u) -> unsigned {
      std::string name(prefix);
      name += symbol.getValue().str();
      if (ordinal != ~0u) {
        name += "_";
        name += std::to_string(ordinal);
      }
      unsigned index = boolVars.size();
      boolVars.push_back(context.bool_const(name.c_str()));
      return index;
    };

    auto addRankVar = [&](StringAttr symbol) -> unsigned {
      std::string name("lp_rank_");
      name += symbol.getValue().str();
      unsigned index = rankVars.size();
      rankVars.push_back(context.int_const(name.c_str()));
      return index;
    };

    for (auto &node : nodes) {
      node.activeVarIndex = addBoolVar("lp_active_", node.symbol);
      node.rankVarIndex = addRankVar(node.symbol);
      for (auto indexedChoice : llvm::enumerate(node.choices))
        indexedChoice.value().boolVarIndex =
            addBoolVar("lp_choice_", node.symbol, indexedChoice.index());
    }

    for (const LpNode &node : nodes) {
      z3::expr active = boolVars[node.activeVarIndex];
      z3::expr rank = rankVars[node.rankVarIndex];
      if (node.isInput) {
        optimizer.add(rank == context.int_val(0));
        continue;
      }

      optimizer.add(rank >= context.int_val(0));
      if (node.choices.empty()) {
        optimizer.add(!active);
        continue;
      }

      z3::expr_vector choiceVars(context);
      for (const LpChoice &choice : node.choices)
        choiceVars.push_back(boolVars[choice.boolVarIndex]);
      optimizer.add(active == z3::mk_or(choiceVars));

      for (unsigned i = 0, e = node.choices.size(); i < e; ++i) {
        z3::expr lhs = boolVars[node.choices[i].boolVarIndex];
        for (unsigned j = i + 1; j < e; ++j) {
          z3::expr rhs = boolVars[node.choices[j].boolVarIndex];
          optimizer.add(!(lhs && rhs));
        }
      }
    }

    for (unsigned rootIndex : rootNodeIndices)
      optimizer.add(boolVars[nodes[rootIndex].activeVarIndex]);

    for (const LpNode &node : nodes) {
      for (const LpChoice &choice : node.choices) {
        z3::expr selected = boolVars[choice.boolVarIndex];
        for (StringAttr childSymbol : choice.childSymbols) {
          auto child = nodeIndices.find(childSymbol);
          if (child == nodeIndices.end())
            return failure();
          const LpNode &childNode = nodes[child->second];
          optimizer.add(
              z3::implies(selected, boolVars[childNode.activeVarIndex]));
          optimizer.add(
              z3::implies(selected, rankVars[node.rankVarIndex] >
                                        rankVars[childNode.rankVarIndex]));
        }
      }
    }

    z3::expr totalCost = context.int_val(0);
    z3::expr activeCount = context.int_val(0);
    z3::expr selectedChoiceCount = context.int_val(0);
    SmallVector<unsigned, 16> orderedChoiceVarIndices;
    for (const LpNode &node : nodes) {
      activeCount =
          activeCount + z3::ite(boolVars[node.activeVarIndex],
                                context.int_val(1), context.int_val(0));
      for (const LpChoice &choice : node.choices) {
        orderedChoiceVarIndices.push_back(choice.boolVarIndex);
        selectedChoiceCount = selectedChoiceCount +
                              z3::ite(boolVars[choice.boolVarIndex],
                                      context.int_val(1), context.int_val(0));
        if (choice.cost == 0)
          continue;
        totalCost = totalCost + z3::ite(boolVars[choice.boolVarIndex],
                                        context.int_val(choice.cost),
                                        context.int_val(0));
      }
    }

    // Keep equal-cost models stable and avoid detached zero-cost selections.
    optimizer.minimize(totalCost);
    optimizer.minimize(activeCount);
    optimizer.minimize(selectedChoiceCount);
    for (unsigned choiceVarIndex : orderedChoiceVarIndices)
      optimizer.minimize(z3::ite(boolVars[choiceVarIndex], context.int_val(0),
                                 context.int_val(1)));

    if (optimizer.check() != z3::sat)
      return failure();

    z3::model model = optimizer.get_model();
    for (const LpNode &node : nodes) {
      bool active = model.eval(boolVars[node.activeVarIndex], true).is_true();
      if (!active || node.isInput)
        continue;

      const LpChoice *selectedChoice = nullptr;
      for (const LpChoice &choice : node.choices) {
        if (!model.eval(boolVars[choice.boolVarIndex], true).is_true())
          continue;
        if (selectedChoice)
          return failure();
        selectedChoice = &choice;
      }
      if (!selectedChoice)
        return failure();

      GreedyExtractSelection selection;
      selection.eclass = graph.getValue(node.symbol);
      if (selectedChoice->kind == LpChoiceKind::Alias) {
        selection.kind = GreedyExtractSelection::Kind::Alias;
        selection.aliasTarget = graph.getValue(selectedChoice->aliasTarget);
      } else {
        selection.kind = GreedyExtractSelection::Kind::Candidate;
        selection.candidateRoot = selectedChoice->candidateRoot;
        selection.cost = selectedChoice->cost;
      }
      selections.insert({node.symbol, selection});
    }

    return success();
  }

  FailureOr<GreedyExtractSelection> lookupSelection(StringAttr symbol) {
    auto selected = selections.find(symbol);
    if (selected != selections.end())
      return selected->second;

    if (egraph.lookupSymbol<InputOp>(symbol.getValue())) {
      GreedyExtractSelection selection;
      selection.eclass = graph.getValue(symbol);
      selection.kind = GreedyExtractSelection::Kind::Input;
      return selection;
    }

    return failure();
  }

  FailureOr<EGraphExtractCost>
  collectSelected(StringAttr symbol, EGraphExtractInfo &result,
                  DenseMap<StringAttr, EGraphExtractCost> &subtreeCosts,
                  SmallVectorImpl<StringAttr> &collectingSymbols) {
    auto costIt = subtreeCosts.find(symbol);
    if (costIt != subtreeCosts.end())
      return costIt->second;

    if (llvm::is_contained(collectingSymbols, symbol))
      return failure();

    FailureOr<GreedyExtractSelection> selection = lookupSelection(symbol);
    if (failed(selection))
      return failure();

    if (selection->kind == GreedyExtractSelection::Kind::Input) {
      subtreeCosts.insert({symbol, 0});
      return EGraphExtractCost(0);
    }

    collectingSymbols.push_back(symbol);
    if (selection->kind == GreedyExtractSelection::Kind::Alias) {
      FailureOr<EGraphExtractCost> targetCost =
          collectSelected(selection->aliasTarget.getSymbolNameAttr(), result,
                          subtreeCosts, collectingSymbols);
      collectingSymbols.pop_back();
      if (failed(targetCost))
        return failure();

      selection->subtreeCost = *targetCost;
      if (!llvm::is_contained(emittedAliasSymbols, symbol)) {
        emittedAliasSymbols.push_back(symbol);
        result.selectedAliasEClasses.push_back(selection->eclass);
        result.selectedAliasTargets.push_back(selection->aliasTarget);
      }
      subtreeCosts.insert({symbol, *targetCost});
      return *targetCost;
    }

    FailureOr<EGraphStructuralKey> key =
        graph.getStructuralKey(selection->candidateRoot);
    if (failed(key)) {
      collectingSymbols.pop_back();
      return failure();
    }

    EGraphExtractCost totalCost = selection->cost;
    for (StringAttr childSymbol : key->childLeaderSymbols) {
      FailureOr<EGraphExtractCost> childCost =
          collectSelected(childSymbol, result, subtreeCosts, collectingSymbols);
      if (failed(childCost)) {
        collectingSymbols.pop_back();
        return failure();
      }

      FailureOr<EGraphExtractCost> updatedCost =
          addExtractCosts(totalCost, *childCost);
      if (failed(updatedCost)) {
        collectingSymbols.pop_back();
        return failure();
      }
      totalCost = *updatedCost;
    }
    collectingSymbols.pop_back();

    selection->subtreeCost = totalCost;
    if (!llvm::is_contained(emittedSymbols, symbol)) {
      emittedSymbols.push_back(symbol);
      result.selectedEClasses.push_back(selection->eclass);
      result.selectedCandidateRoots.push_back(selection->candidateRoot);
      result.selectedCandidateCosts.push_back(selection->cost);
      result.selectedSubtreeCosts.push_back(totalCost);
    }
    subtreeCosts.insert({symbol, totalCost});
    return totalCost;
  }

  EGraph &graph;
  EGraphOp egraph;
  EGraphExtractCostModel costModel;
  SmallVector<LpNode, 8> nodes;
  DenseMap<StringAttr, unsigned> nodeIndices;
  SmallVector<unsigned, 4> rootNodeIndices;
  DenseMap<StringAttr, GreedyExtractSelection> selections;
  SmallVector<StringAttr, 4> emittedSymbols;
  SmallVector<StringAttr, 4> emittedAliasSymbols;
};
#endif // MLIR_EGRAPH_ENABLE_Z3

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

FailureOr<EGraphExtractRequest>
buildRequestFromRoots(EGraph &graph, EGraphOp egraph,
                      ArrayRef<EValue> rootsToNormalize,
                      EGraphExtractMode mode) {
  ArrayRef<Type> resultTypes = egraph.getResultTypes();
  if (rootsToNormalize.size() != resultTypes.size())
    return failure();

  EGraphExtractRequest request;
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
} // namespace

StringRef mlir::egraph::stringifyEGraphExtractMode(EGraphExtractMode mode) {
  switch (mode) {
  case EGraphExtractMode::Greedy:
    return "greedy";
  case EGraphExtractMode::LinearProgramming:
    return "lp";
  }
  llvm_unreachable("unexpected egraph extract mode");
}

static FailureOr<EGraphExtractRequest>
buildExtractRequest(EGraph &graph, EGraphOp egraph,
                    ArrayRef<EValue> explicitRoots,
                    EGraphExtractMode mode) {
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

static FailureOr<EGraphExtractInfo>
runGreedyExtract(EGraph &graph, EGraphOp egraph,
                 const EGraphExtractRequest &request,
                 EGraphExtractCostModel costModel) {
  GreedyEGraphExtractor extractor(graph, egraph, costModel);
  return extractor.run(request);
}

static FailureOr<EGraphExtractInfo>
runLinearProgrammingExtract(EGraph &graph, EGraphOp egraph,
                            const EGraphExtractRequest &request,
                            EGraphExtractCostModel costModel) {
#ifdef MLIR_EGRAPH_ENABLE_Z3
  LinearProgrammingEGraphExtractor extractor(graph, egraph, costModel);
  return extractor.run(request);
#else
  (void)graph;
  (void)egraph;
  (void)request;
  (void)costModel;
  return failure();
#endif
}

LogicalResult mlir::egraph::extractEGraphForTesting(
    EGraph &graph, EGraphOp egraph, EGraphExtractMode mode,
    EGraphExtractCostModel costModel, EGraphExtractInfo *info,
    ArrayRef<EValue> explicitRoots) {
  FailureOr<EGraphExtractRequest> request =
      buildExtractRequest(graph, egraph, explicitRoots, mode);
  if (failed(request))
    return failure();

  FailureOr<EGraphExtractInfo> selection;
  switch (mode) {
  case EGraphExtractMode::Greedy:
    selection = runGreedyExtract(graph, egraph, *request, costModel);
    break;
  case EGraphExtractMode::LinearProgramming:
    selection = runLinearProgrammingExtract(graph, egraph, *request, costModel);
    break;
  }
  if (failed(selection))
    return failure();

  if (info)
    *info = *selection;
  return success();
}

FailureOr<SmallVector<Value, 4>>
mlir::egraph::materializeEGraphExtractInfoForTesting(
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
  FailureOr<EGraphExtractRequest> request =
      buildExtractRequest(*state.graph, egraph, explicitRoots, mode);
  if (failed(request))
    return failure();

  FailureOr<EGraphExtractInfo> selection;
  switch (mode) {
  case EGraphExtractMode::Greedy:
    selection = runGreedyExtract(*state.graph, egraph, *request, costModel);
    break;
  case EGraphExtractMode::LinearProgramming:
    selection = runLinearProgrammingExtract(*state.graph, egraph, *request,
                                            costModel);
    break;
  }
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
  FailureOr<SmallVector<Value, 4>> roots =
      materializeExtractSelection(*state.graph, egraph, *selection, builder,
                                 inputValues);
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
