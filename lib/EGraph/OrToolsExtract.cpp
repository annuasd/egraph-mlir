#include "ExtractInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#ifdef MLIR_EGRAPH_ENABLE_OR_TOOLS
#include "ortools/sat/cp_model.h"
#endif

namespace mlir::egraph::detail {

#ifdef MLIR_EGRAPH_ENABLE_OR_TOOLS
namespace {
FailureOr<int64_t> addCheckedInt64(int64_t lhs, int64_t rhs) {
  if (rhs > std::numeric_limits<int64_t>::max() - lhs)
    return failure();
  return lhs + rhs;
}

FailureOr<int64_t> convertCostToCpSat(EGraphExtractCost cost) {
  if (cost >
      static_cast<EGraphExtractCost>(std::numeric_limits<int64_t>::max()))
    return failure();
  return static_cast<int64_t>(cost);
}

class OrToolsLinearProgrammingEGraphExtractor {
public:
  OrToolsLinearProgrammingEGraphExtractor(EGraph &graph, EGraphOp egraph,
                                          EGraphExtractCostModel costModel)
      : graph(graph), egraph(egraph), costModel(costModel) {}

  FailureOr<EGraphExtractInfo> run(const EGraphExtractRequest &request) {
    if (!graph.isClean() || !egraph ||
        request.mode != EGraphExtractMode::OrToolsLinearProgramming)
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
    result.mode = EGraphExtractMode::OrToolsLinearProgramming;
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
    operations_research::sat::BoolVar boolVar;
  };

  struct LpNode {
    StringAttr symbol;
    bool isInput = false;
    bool populated = false;
    bool populating = false;
    operations_research::sat::BoolVar activeVar;
    operations_research::sat::IntVar rankVar;
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
    using operations_research::Domain;
    using operations_research::sat::BoolVar;
    using operations_research::sat::CpModelBuilder;
    using operations_research::sat::CpSolverResponse;
    using operations_research::sat::CpSolverStatus;
    using operations_research::sat::LinearExpr;
    using operations_research::sat::SatParameters;
    using operations_research::sat::SolutionBooleanValue;
    using operations_research::sat::SolutionIntegerValue;
    using operations_research::sat::SolveWithParameters;

    CpModelBuilder cpModel;
    cpModel.SetName("mlir_egraph_extract");

    const int64_t maxRank = static_cast<int64_t>(nodes.size());
    std::vector<BoolVar> activeVars;
    std::vector<BoolVar> orderedChoiceVars;
    activeVars.reserve(nodes.size());

    auto makeVarName = [](StringRef prefix, StringAttr symbol,
                          unsigned ordinal = ~0u) {
      std::string name(prefix);
      name += symbol.getValue().str();
      if (ordinal != ~0u) {
        name += "_";
        name += std::to_string(ordinal);
      }
      return name;
    };

    for (LpNode &node : nodes) {
      node.activeVar =
          cpModel.NewBoolVar().WithName(makeVarName("lp_active_", node.symbol));
      node.rankVar = cpModel.NewIntVar(Domain(0, maxRank))
                         .WithName(makeVarName("lp_rank_", node.symbol));
      activeVars.push_back(node.activeVar);
      for (auto indexedChoice : llvm::enumerate(node.choices)) {
        indexedChoice.value().boolVar = cpModel.NewBoolVar().WithName(
            makeVarName("lp_choice_", node.symbol, indexedChoice.index()));
        orderedChoiceVars.push_back(indexedChoice.value().boolVar);
      }
    }

    for (const LpNode &node : nodes) {
      if (node.isInput) {
        cpModel.AddEquality(node.rankVar, 0);
        continue;
      }

      std::vector<BoolVar> choiceVars;
      choiceVars.reserve(node.choices.size());
      for (const LpChoice &choice : node.choices)
        choiceVars.push_back(choice.boolVar);
      cpModel.AddEquality(LinearExpr::Sum(choiceVars), node.activeVar);
    }

    for (unsigned rootIndex : rootNodeIndices)
      cpModel.FixVariable(nodes[rootIndex].activeVar, true);

    for (const LpNode &node : nodes) {
      for (const LpChoice &choice : node.choices) {
        for (StringAttr childSymbol : choice.childSymbols) {
          auto child = nodeIndices.find(childSymbol);
          if (child == nodeIndices.end())
            return failure();

          const LpNode &childNode = nodes[child->second];
          cpModel.AddImplication(choice.boolVar, childNode.activeVar);
          cpModel.AddGreaterThan(node.rankVar, childNode.rankVar)
              .OnlyEnforceIf(choice.boolVar);
        }
      }
    }

    LinearExpr totalCost;
    int64_t totalCostUpperBound = 0;
    for (const LpNode &node : nodes) {
      for (const LpChoice &choice : node.choices) {
        FailureOr<int64_t> cost = convertCostToCpSat(choice.cost);
        if (failed(cost))
          return failure();
        FailureOr<int64_t> updatedUpperBound =
            addCheckedInt64(totalCostUpperBound, *cost);
        if (failed(updatedUpperBound))
          return failure();
        totalCostUpperBound = *updatedUpperBound;
        if (*cost == 0)
          continue;
        totalCost += LinearExpr::Term(choice.boolVar, *cost);
      }
    }
    LinearExpr activeCount = LinearExpr::Sum(activeVars);
    LinearExpr selectedChoiceCount = LinearExpr::Sum(orderedChoiceVars);

    CpSolverResponse response;
    SatParameters parameters;
    parameters.set_num_search_workers(1);

    auto solveAndConstrainObjective =
        [&](const LinearExpr &objective) -> FailureOr<int64_t> {
      cpModel.ClearObjective();
      cpModel.Minimize(objective);
      response = SolveWithParameters(cpModel.Build(), parameters);
      if (response.status() != CpSolverStatus::OPTIMAL)
        return failure();

      int64_t objectiveValue = SolutionIntegerValue(response, objective);
      cpModel.AddEquality(objective, objectiveValue);
      return objectiveValue;
    };

    if (failed(solveAndConstrainObjective(totalCost)) ||
        failed(solveAndConstrainObjective(activeCount)) ||
        failed(solveAndConstrainObjective(selectedChoiceCount)))
      return failure();

    for (BoolVar choiceVar : orderedChoiceVars) {
      cpModel.ClearObjective();
      cpModel.Minimize(LinearExpr(choiceVar.Not()));
      response = SolveWithParameters(cpModel.Build(), parameters);
      if (response.status() != CpSolverStatus::OPTIMAL)
        return failure();

      bool selected = SolutionBooleanValue(response, choiceVar);
      cpModel.FixVariable(choiceVar, selected);
    }

    for (const LpNode &node : nodes) {
      bool active = SolutionBooleanValue(response, node.activeVar);
      if (!active || node.isInput)
        continue;

      const LpChoice *selectedChoice = nullptr;
      for (const LpChoice &choice : node.choices) {
        if (!SolutionBooleanValue(response, choice.boolVar))
          continue;
        if (selectedChoice)
          return failure();
        selectedChoice = &choice;
      }
      if (!selectedChoice)
        return failure();

      ExtractSelection selection;
      selection.eclass = graph.getValue(node.symbol);
      if (selectedChoice->kind == LpChoiceKind::Alias) {
        selection.kind = ExtractSelection::Kind::Alias;
        selection.aliasTarget = graph.getValue(selectedChoice->aliasTarget);
      } else {
        selection.kind = ExtractSelection::Kind::Candidate;
        selection.candidateRoot = selectedChoice->candidateRoot;
        selection.cost = selectedChoice->cost;
      }
      selections.insert({node.symbol, selection});
    }

    return success();
  }

  FailureOr<ExtractSelection> lookupSelection(StringAttr symbol) {
    auto selected = selections.find(symbol);
    if (selected != selections.end())
      return selected->second;

    if (egraph.lookupSymbol<InputOp>(symbol.getValue())) {
      ExtractSelection selection;
      selection.eclass = graph.getValue(symbol);
      selection.kind = ExtractSelection::Kind::Input;
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

    FailureOr<ExtractSelection> selection = lookupSelection(symbol);
    if (failed(selection))
      return failure();

    if (selection->kind == ExtractSelection::Kind::Input) {
      subtreeCosts.insert({symbol, 0});
      return EGraphExtractCost(0);
    }

    collectingSymbols.push_back(symbol);
    if (selection->kind == ExtractSelection::Kind::Alias) {
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
  DenseMap<StringAttr, ExtractSelection> selections;
  SmallVector<StringAttr, 4> emittedSymbols;
  SmallVector<StringAttr, 4> emittedAliasSymbols;
};
} // namespace
#endif // MLIR_EGRAPH_ENABLE_OR_TOOLS

FailureOr<EGraphExtractInfo>
runOrToolsLinearProgrammingExtract(EGraph &graph, EGraphOp egraph,
                                   const EGraphExtractRequest &request,
                                   EGraphExtractCostModel costModel) {
#ifdef MLIR_EGRAPH_ENABLE_OR_TOOLS
  OrToolsLinearProgrammingEGraphExtractor extractor(graph, egraph, costModel);
  return extractor.run(request);
#else
  (void)graph;
  (void)egraph;
  (void)request;
  (void)costModel;
  return failure();
#endif
}

} // namespace mlir::egraph::detail
