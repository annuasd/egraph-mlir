#include "ExtractInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <string>
#include <vector>

#ifdef MLIR_EGRAPH_ENABLE_Z3
#include <z3++.h>
#endif

namespace mlir::egraph::detail {

#ifdef MLIR_EGRAPH_ENABLE_Z3
namespace {
class Z3LinearProgrammingEGraphExtractor {
public:
  Z3LinearProgrammingEGraphExtractor(EGraph &graph, EGraphOp egraph,
                                     EGraphExtractCostModel costModel)
      : graph(graph), egraph(egraph), costModel(costModel) {}

  FailureOr<EGraphExtractInfo> run(const EGraphExtractRequest &request) {
    if (!graph.isClean() || !egraph ||
        request.mode != EGraphExtractMode::Z3LinearProgramming)
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
    result.mode = EGraphExtractMode::Z3LinearProgramming;
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
#endif // MLIR_EGRAPH_ENABLE_Z3

FailureOr<EGraphExtractInfo>
runZ3LinearProgrammingExtract(EGraph &graph, EGraphOp egraph,
                              const EGraphExtractRequest &request,
                              EGraphExtractCostModel costModel) {
#ifdef MLIR_EGRAPH_ENABLE_Z3
  Z3LinearProgrammingEGraphExtractor extractor(graph, egraph, costModel);
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
