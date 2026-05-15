#include "ExtractInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::egraph::detail {
namespace {
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

    SmallVector<ExtractSelection, 4> rootSelections;
    rootSelections.reserve(request.roots.size());

    for (EValue root : request.roots) {
      FailureOr<ExtractSelection> selection = computeBest(root);
      if (failed(selection))
        return failure();
      rootSelections.push_back(*selection);
    }

    for (const ExtractSelection &selection : rootSelections)
      result.rootCosts.push_back(selection.subtreeCost);

    for (const ExtractSelection &selection : rootSelections)
      if (failed(collectSelected(selection.eclass, result)))
        return failure();

    result.changed = !result.selectedCandidateRoots.empty() ||
                     !result.selectedAliasEClasses.empty();
    return result;
  }

private:
  FailureOr<ExtractSelection> computeBest(EValue root) {
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
    FailureOr<ExtractSelection> selection = computeBestUncached(leader);
    activeSymbols.pop_back();
    if (failed(selection))
      return failure();

    bestBySymbol.insert({symbol, *selection});
    return *selection;
  }

  FailureOr<ExtractSelection> computeBestPreservingSymbol(EValue root) {
    if (!root || root.getGraph() != &graph || !root.getSymbolNameAttr())
      return failure();

    StringAttr symbol = root.getSymbolNameAttr();
    auto cached = rawBestBySymbol.find(symbol);
    if (cached != rawBestBySymbol.end())
      return cached->second;

    if (llvm::is_contained(activeRawSymbols, symbol))
      return failure();

    activeRawSymbols.push_back(symbol);
    FailureOr<ExtractSelection> selection =
        computeBestUncachedPreservingSymbol(root);
    activeRawSymbols.pop_back();
    if (failed(selection))
      return failure();

    rawBestBySymbol.insert({symbol, *selection});
    return *selection;
  }

  FailureOr<ExtractSelection> computeBestUncached(EValue leader) {
    bool hasBestCandidate = false;
    ExtractSelection bestSelection;

    if (auto input = egraph.lookupSymbol<InputOp>(leader.getSymbolName())) {
      (void)input;
      return computeInputLeaf(leader);
    }

    auto eclass = egraph.lookupSymbol<EClassOp>(leader.getSymbolName());
    if (!eclass)
      return failure();

    SmallVector<EOpRefBase, 4> candidates = graph.getCandidateRoots(eclass);
    for (EOpRefBase candidate : candidates) {
      FailureOr<ExtractSelection> candidateSelection =
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

      FailureOr<ExtractSelection> aliasSelection =
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

  FailureOr<ExtractSelection> computeBestUncachedPreservingSymbol(EValue root) {
    bool hasBestCandidate = false;
    ExtractSelection bestSelection;

    if (auto input = egraph.lookupSymbol<InputOp>(root.getSymbolName())) {
      (void)input;
      return computeInputLeaf(root);
    }

    auto eclass = egraph.lookupSymbol<EClassOp>(root.getSymbolName());
    if (!eclass)
      return failure();

    SmallVector<EOpRefBase, 4> candidates = graph.getCandidateRoots(eclass);
    for (EOpRefBase candidate : candidates) {
      FailureOr<ExtractSelection> candidateSelection =
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

      FailureOr<ExtractSelection> aliasSelection =
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

  FailureOr<ExtractSelection> computeAliasSelection(EClassOp eclass,
                                                    unsigned candidateOrdinal) {
    FailureOr<StringAttr> aliasTarget =
        getAliasCandidateTarget(eclass, candidateOrdinal);
    if (failed(aliasTarget))
      return failure();

    FailureOr<ExtractSelection> child =
        computeBestPreservingSymbol(graph.getValue(*aliasTarget));
    if (failed(child))
      return failure();

    ExtractSelection selection;
    selection.eclass = graph.getValue(eclass.getSymNameAttr());
    selection.aliasTarget = child->eclass;
    selection.kind = ExtractSelection::Kind::Alias;
    selection.subtreeCost = child->subtreeCost;
    return selection;
  }

  FailureOr<ExtractSelection> computeInputLeaf(EValue leader) {
    if (!egraph.lookupSymbol<InputOp>(leader.getSymbolName()))
      return failure();

    ExtractSelection selection;
    selection.eclass = leader;
    selection.kind = ExtractSelection::Kind::Input;
    return selection;
  }

  FailureOr<ExtractSelection> computeCandidateSelection(EOpRefBase candidate) {
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

      FailureOr<ExtractSelection> child =
          computeBest(graph.getValue(childSymbol));
      if (failed(child))
        return failure();

      FailureOr<EGraphExtractCost> updatedCost =
          addExtractCosts(totalCost, child->subtreeCost);
      if (failed(updatedCost))
        return failure();
      totalCost = *updatedCost;
    }

    ExtractSelection selection;
    selection.eclass = candidate.getResult(0).getLeader();
    selection.candidateRoot = candidate;
    selection.cost = *localCost;
    selection.subtreeCost = totalCost;
    selection.kind = ExtractSelection::Kind::Candidate;
    return selection;
  }

  LogicalResult collectSelected(EValue root, EGraphExtractInfo &result,
                                bool preserveSymbol = false) {
    FailureOr<ExtractSelection> selection =
        preserveSymbol ? computeBestPreservingSymbol(root) : computeBest(root);
    if (failed(selection))
      return failure();

    if (selection->kind == ExtractSelection::Kind::Input)
      return success();

    StringAttr symbol = selection->eclass.getSymbolNameAttr();
    if (selection->kind == ExtractSelection::Kind::Alias) {
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
  llvm::DenseMap<StringAttr, ExtractSelection> bestBySymbol;
  llvm::DenseMap<StringAttr, ExtractSelection> rawBestBySymbol;
  SmallVector<StringAttr, 4> activeSymbols;
  SmallVector<StringAttr, 4> activeRawSymbols;
  SmallVector<StringAttr, 4> collectingSymbols;
  SmallVector<StringAttr, 4> emittedSymbols;
  SmallVector<StringAttr, 4> emittedAliasSymbols;
};
} // namespace

FailureOr<EGraphExtractInfo>
runGreedyExtract(EGraph &graph, EGraphOp egraph,
                 const EGraphExtractRequest &request,
                 EGraphExtractCostModel costModel) {
  GreedyEGraphExtractor extractor(graph, egraph, costModel);
  return extractor.run(request);
}

} // namespace mlir::egraph::detail
