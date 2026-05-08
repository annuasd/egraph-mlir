#define MLIR_EGRAPH_ENABLE_TEST_DIALECT
#define MLIR_EGRAPH_TEST_LIBRARY

#include "MLIREGraph/IR/EGraphDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#ifdef MLIR_EGRAPH_ENABLE_TEST_DIALECT
void registerEGraphTestPasses();
#endif

#ifdef MLIR_EGRAPH_ENABLE_TEST_DIALECT
#include "EGraphTest/IR/EGraphTestDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#endif

#if defined(MLIR_EGRAPH_ENABLE_TEST_DIALECT) &&                                \
    defined(MLIR_EGRAPH_TEST_LIBRARY)
#include "EGraphTest/IR/EGraphTestOps.h"
#include "MLIREGraph/EGraph/EGraph.h"
#include "MLIREGraph/EGraph/Extract.h"
#include "MLIREGraph/EGraph/Pattern.h"
#include "mlir/IR/Builders.h"
#endif

#if defined(MLIR_EGRAPH_ENABLE_TEST_DIALECT) &&                                \
    defined(MLIR_EGRAPH_TEST_LIBRARY)
namespace mlir::egraph {
mlir::FailureOr<EGraphMatchStats>
applyEGraphPatterns(mlir::egraph::EGraph &graph,
                    const EGraphPatternSet &patterns,
                    mlir::Operation *rebuildRoot,
                    const EGraphMatchConfig &config);
mlir::FailureOr<EGraphMatchStats>
applyEGraphPatterns(mlir::egraph::EGraph &graph,
                    const EGraphPatternSet &patterns,
                    mlir::Operation *rebuildRoot);
mlir::LogicalResult extractEGraphForTesting(
    mlir::egraph::EGraph &graph, EGraphOp egraph, EGraphExtractMode mode,
    EGraphExtractCostModel costModel, EGraphExtractInfo *info,
    llvm::ArrayRef<mlir::egraph::EValue> explicitRoots = {});
mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
materializeEGraphExtractInfoForTesting(
    mlir::egraph::EGraph &graph, EGraphOp egraph,
    const EGraphExtractInfo &selection, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::Value> inputValues);
} // namespace mlir::egraph

namespace {

mlir::LogicalResult indexTestGraph(mlir::Operation *root,
                                   mlir::egraph::EGraph &graph) {
  auto module = mlir::dyn_cast<mlir::ModuleOp>(root);
  if (!module)
    return mlir::failure();

  mlir::egraph::EGraphOp symbolicGraph;
  for (mlir::egraph::EGraphOp egraph :
       module.getOps<mlir::egraph::EGraphOp>()) {
    if (symbolicGraph)
      return module.emitError("expected at most one active egraph.egraph");
    symbolicGraph = egraph;
  }

  if (symbolicGraph)
    return graph.indexEGraph(symbolicGraph);
  return module.emitError("expected one active egraph.egraph");
}

mlir::FailureOr<mlir::egraph::EGraphOp>
getSingleActiveEGraph(mlir::ModuleOp module) {
  mlir::egraph::EGraphOp symbolicGraph;
  for (mlir::egraph::EGraphOp egraph :
       module.getOps<mlir::egraph::EGraphOp>()) {
    if (symbolicGraph)
      return mlir::failure();
    symbolicGraph = egraph;
  }
  if (!symbolicGraph)
    return mlir::failure();
  return symbolicGraph;
}

mlir::LogicalResult seedDirtyWorklistGraph(mlir::ModuleOp module,
                                           mlir::egraph::EGraph &graph) {
  mlir::FailureOr<mlir::egraph::EGraphOp> egraph =
      getSingleActiveEGraph(module);
  if (mlir::failed(egraph))
    return module.emitError("expected one active egraph.egraph");

  auto lhs = egraph->lookupSymbol<mlir::egraph::EClassOp>("lhs");
  auto rhs = egraph->lookupSymbol<mlir::egraph::EClassOp>("rhs");
  if (!lhs || !rhs)
    return egraph->emitOpError("expected @lhs and @rhs symbols");

  mlir::FailureOr<mlir::egraph::EGraphUnionResult> unioned =
      graph.unionValues(graph.getValue(lhs.getSymNameAttr()),
                        graph.getValue(rhs.getSymNameAttr()));
  if (mlir::failed(unioned) || !unioned->changed)
    return egraph->emitOpError("failed to seed a dirty symbolic graph");
  if (graph.isClean())
    return egraph->emitOpError(
        "symbolic union unexpectedly left the graph clean");
  return mlir::success();
}

void emitDriverStatsRemark(
    mlir::Operation *anchor,
    const mlir::egraph::EGraphMatchStats &result) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  mlir::egraph::printEGraphMatchStats(os, result);
  os.flush();
  anchor->emitRemark() << "worklist driver stats: " << buffer;
}

std::string formatExtractRoots(llvm::ArrayRef<mlir::egraph::EValue> roots) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  if (roots.empty()) {
    os << "<none>";
  } else {
    llvm::interleaveComma(roots, os, [&](mlir::egraph::EValue root) {
      if (mlir::StringAttr symbol = root.getSymbolNameAttr())
        os << '@' << symbol.getValue();
      else
        os << "<invalid>";
    });
  }
  os.flush();
  return buffer;
}

mlir::LogicalResult
verifyExtractRoots(mlir::Operation *anchor,
                   llvm::ArrayRef<mlir::egraph::EValue> roots,
                   llvm::ArrayRef<mlir::StringAttr> expectedRoots) {
  if (roots.size() != expectedRoots.size())
    return anchor->emitOpError("extract returned ")
           << roots.size() << " root(s), expected " << expectedRoots.size();

  for (auto indexedRoot : llvm::enumerate(roots)) {
    mlir::egraph::EValue root = indexedRoot.value();
    if (root.getResultIndex() != 0)
      return anchor->emitOpError("extract root #")
             << indexedRoot.index() << " did not use result slot 0";
    if (root.getSymbolNameAttr() != expectedRoots[indexedRoot.index()])
      return anchor->emitOpError("extract root #")
             << indexedRoot.index() << " was @"
             << root.getSymbolNameAttr().getValue() << ", expected @"
             << expectedRoots[indexedRoot.index()].getValue();
  }

  return mlir::success();
}

mlir::LogicalResult verifyExtractInfoSemantics(mlir::ModuleOp module,
                                               mlir::egraph::EGraph &graph) {
  mlir::FailureOr<mlir::egraph::EGraphOp> egraph =
      getSingleActiveEGraph(module);
  if (mlir::failed(egraph))
    return module.emitError("expected one active egraph.egraph");

  auto lhs = egraph->lookupSymbol<mlir::egraph::EClassOp>("lhs");
  auto rhs = egraph->lookupSymbol<mlir::egraph::EClassOp>("rhs");
  auto user = egraph->lookupSymbol<mlir::egraph::EClassOp>("user");
  if (!lhs || !rhs || !user)
    return egraph->emitOpError("expected @lhs, @rhs, and @user symbols");

  auto costModel = [](mlir::Operation *)
      -> mlir::FailureOr<mlir::egraph::EGraphExtractCost> {
    return mlir::egraph::EGraphExtractCost(0);
  };

  mlir::egraph::EGraphExtractInfo defaultInfo;
  if (mlir::failed(mlir::egraph::extractEGraphForTesting(
          graph, *egraph, mlir::egraph::EGraphExtractMode::Greedy, costModel,
          &defaultInfo)))
    return egraph->emitOpError("failed to extract default info");
  if (defaultInfo.mode != mlir::egraph::EGraphExtractMode::Greedy)
    return egraph->emitOpError("default extract info used a wrong mode");
  llvm::SmallVector<mlir::StringAttr, 4> defaultRoots = {rhs.getSymNameAttr(),
                                                         user.getSymNameAttr()};
  if (mlir::failed(
          verifyExtractRoots(egraph->getOperation(), defaultInfo.roots,
                             defaultRoots)))
    return mlir::failure();

  egraph->emitRemark() << "extract info default roots -> "
                       << formatExtractRoots(defaultInfo.roots) << " mode="
                       << mlir::egraph::stringifyEGraphExtractMode(
                              defaultInfo.mode);

  llvm::SmallVector<mlir::egraph::EValue, 4> explicitRoots = {
      graph.getValue(lhs.getSymNameAttr()),
      graph.getValue(user.getSymNameAttr())};
  mlir::egraph::EGraphExtractInfo explicitInfo;
  if (mlir::failed(mlir::egraph::extractEGraphForTesting(
          graph, *egraph, mlir::egraph::EGraphExtractMode::LinearProgramming,
          costModel, &explicitInfo, explicitRoots)))
    return egraph->emitOpError("failed to extract explicit info");
  if (explicitInfo.mode !=
      mlir::egraph::EGraphExtractMode::LinearProgramming)
    return egraph->emitOpError("explicit extract info lost its mode");
  llvm::SmallVector<mlir::StringAttr, 4> explicitExpectedRoots = {
      lhs.getSymNameAttr(), user.getSymNameAttr()};
  if (mlir::failed(verifyExtractRoots(egraph->getOperation(),
                                      explicitInfo.roots,
                                      explicitExpectedRoots)))
    return mlir::failure();

  egraph->emitRemark() << "extract info explicit roots -> "
                       << formatExtractRoots(explicitInfo.roots) << " mode="
                       << mlir::egraph::stringifyEGraphExtractMode(
                              explicitInfo.mode);

  mlir::FailureOr<mlir::egraph::EGraphUnionResult> unioned =
      graph.unionValues(graph.getValue(lhs.getSymNameAttr()),
                        graph.getValue(rhs.getSymNameAttr()));
  if (mlir::failed(unioned) || !unioned->changed)
    return egraph->emitOpError("failed to dirty extract info graph");
  if (graph.isClean())
    return egraph->emitOpError("dirty union unexpectedly left graph clean");
  mlir::egraph::EGraphExtractInfo dirtyInfo;
  if (mlir::succeeded(mlir::egraph::extractEGraphForTesting(
          graph, *egraph, mlir::egraph::EGraphExtractMode::Greedy, costModel,
          &dirtyInfo)))
    return egraph->emitOpError(
        "extract unexpectedly accepted a dirty graph");

  egraph->emitRemark() << "extract info rejected dirty graph";
  return mlir::success();
}

mlir::LogicalResult
verifyExtractCostModelSelection(mlir::ModuleOp module,
                                mlir::egraph::EGraph &graph) {
  mlir::FailureOr<mlir::egraph::EGraphOp> egraph =
      getSingleActiveEGraph(module);
  if (mlir::failed(egraph))
    return module.emitError("expected one active egraph.egraph");

  auto costModel = [](mlir::Operation *candidate)
      -> mlir::FailureOr<mlir::egraph::EGraphExtractCost> {
    if (candidate->getName().getStringRef() == "arith.muli")
      return mlir::egraph::EGraphExtractCost(1);
    if (candidate->getName().getStringRef() == "arith.addi")
      return mlir::egraph::EGraphExtractCost(7);
    return mlir::failure();
  };

  mlir::egraph::EGraphExtractInfo selection;
  if (mlir::failed(mlir::egraph::extractEGraphForTesting(
          graph, *egraph, mlir::egraph::EGraphExtractMode::LinearProgramming,
          costModel, &selection)))
    return egraph->emitOpError("failed to select extract candidates");

  if (selection.roots.size() != 1 ||
      selection.selectedCandidateRoots.size() !=
          selection.selectedCandidateCosts.size() ||
      selection.selectedCandidateRoots.size() !=
          selection.selectedSubtreeCosts.size() ||
      selection.selectedEClasses.size() != selection.roots.size() ||
      selection.rootCosts.size() != selection.roots.size())
    return egraph->emitOpError("unexpected extract selection shape");

  if (selection.selectedCandidateRoots.front().getOperationName() !=
          "arith.muli" ||
      selection.selectedCandidateCosts.front() != 1 ||
      selection.selectedSubtreeCosts.front() != 1 ||
      selection.rootCosts.front() != 1)
    return egraph->emitOpError("extract cost model selected the wrong root");

  egraph->emitRemark()
      << "extract cost selection -> " << formatExtractRoots(selection.roots)
      << " candidate="
      << selection.selectedCandidateRoots.front().getOperationName()
      << " cost=" << selection.selectedCandidateCosts.front()
      << " subtree=" << selection.selectedSubtreeCosts.front()
      << " root_cost=" << selection.rootCosts.front()
      << " mode=" << mlir::egraph::stringifyEGraphExtractMode(selection.mode);
  return mlir::success();
}

mlir::FailureOr<unsigned>
findExtractSelection(const mlir::egraph::EGraphExtractInfo &selection,
                     llvm::StringRef symbolName) {
  for (auto indexedEClass : llvm::enumerate(selection.selectedEClasses)) {
    mlir::egraph::EValue eclass = indexedEClass.value();
    if (eclass.getSymbolName() == symbolName)
      return indexedEClass.index();
  }
  return mlir::failure();
}

mlir::LogicalResult
verifyExtractSelectionOrder(mlir::Operation *anchor,
                            const mlir::egraph::EGraphExtractInfo &selection,
                            llvm::ArrayRef<llvm::StringRef> expectedSymbols,
                            llvm::StringRef selectionKind) {
  if (selection.selectedEClasses.size() != expectedSymbols.size())
    return anchor->emitOpError(selectionKind)
           << " extract selected " << selection.selectedEClasses.size()
           << " eclass(es), expected " << expectedSymbols.size();

  for (auto indexedEClass : llvm::enumerate(selection.selectedEClasses)) {
    mlir::egraph::EValue eclass = indexedEClass.value();
    if (eclass.getSymbolName() != expectedSymbols[indexedEClass.index()])
      return anchor->emitOpError(selectionKind)
             << " extract selected eclass #" << indexedEClass.index()
             << " was @" << eclass.getSymbolName() << ", expected @"
             << expectedSymbols[indexedEClass.index()];
  }

  anchor->emitRemark() << selectionKind << " extract selected order -> "
                       << formatExtractRoots(selection.selectedEClasses);
  return mlir::success();
}

mlir::LogicalResult verifyExtractAliasSelection(
    mlir::Operation *anchor, const mlir::egraph::EGraphExtractInfo &selection,
    llvm::ArrayRef<llvm::StringRef> expectedAliasSymbols,
    llvm::ArrayRef<llvm::StringRef> expectedAliasTargets,
    llvm::StringRef selectionKind) {
  if (selection.selectedAliasEClasses.size() != expectedAliasSymbols.size() ||
      selection.selectedAliasTargets.size() != expectedAliasTargets.size())
    return anchor->emitOpError(selectionKind)
           << " extract alias selection shape mismatch";

  for (auto indexedAlias : llvm::enumerate(selection.selectedAliasEClasses)) {
    mlir::egraph::EValue alias = indexedAlias.value();
    mlir::egraph::EValue target =
        selection.selectedAliasTargets[indexedAlias.index()];
    if (alias.getSymbolName() != expectedAliasSymbols[indexedAlias.index()])
      return anchor->emitOpError(selectionKind)
             << " extract alias #" << indexedAlias.index() << " was @"
             << alias.getSymbolName() << ", expected @"
             << expectedAliasSymbols[indexedAlias.index()];
    if (target.getSymbolName() != expectedAliasTargets[indexedAlias.index()])
      return anchor->emitOpError(selectionKind)
             << " extract alias #" << indexedAlias.index() << " targeted @"
             << target.getSymbolName() << ", expected @"
             << expectedAliasTargets[indexedAlias.index()];
  }

  anchor->emitRemark() << selectionKind << " extract alias order -> "
                       << formatExtractRoots(selection.selectedAliasEClasses);
  return mlir::success();
}

mlir::LogicalResult verifyExtractSelectedEClass(
    mlir::Operation *anchor, const mlir::egraph::EGraphExtractInfo &selection,
    llvm::StringRef symbolName, llvm::StringRef operationName,
    mlir::egraph::EGraphExtractCost expectedLocalCost,
    mlir::egraph::EGraphExtractCost expectedSubtreeCost,
    llvm::StringRef selectionKind) {
  mlir::FailureOr<unsigned> index = findExtractSelection(selection, symbolName);
  if (mlir::failed(index))
    return anchor->emitOpError(selectionKind)
           << " extract did not select @" << symbolName;

  if (*index >= selection.selectedCandidateRoots.size() ||
      *index >= selection.selectedCandidateCosts.size() ||
      *index >= selection.selectedSubtreeCosts.size())
    return anchor->emitOpError(selectionKind)
           << " extract selection vectors diverged";

  mlir::egraph::EOpRefBase candidate = selection.selectedCandidateRoots[*index];
  if (candidate.getOperationName() != operationName)
    return anchor->emitOpError(selectionKind)
           << " extract selected @" << symbolName << " as "
           << candidate.getOperationName() << ", expected " << operationName;

  mlir::egraph::EGraphExtractCost actualCost =
      selection.selectedCandidateCosts[*index];
  mlir::egraph::EGraphExtractCost actualSubtreeCost =
      selection.selectedSubtreeCosts[*index];
  if (actualCost != expectedLocalCost)
    return anchor->emitOpError(selectionKind)
           << " extract selected @" << symbolName << " with local cost "
           << actualCost << ", expected " << expectedLocalCost;
  if (actualSubtreeCost != expectedSubtreeCost)
    return anchor->emitOpError(selectionKind)
           << " extract selected @" << symbolName << " with subtree cost "
           << actualSubtreeCost << ", expected " << expectedSubtreeCost;

  anchor->emitRemark() << selectionKind << " extract selected @" << symbolName
                       << " candidate=" << operationName
                       << " local_cost=" << actualCost
                       << " subtree_cost=" << actualSubtreeCost;
  return mlir::success();
}

mlir::LogicalResult
verifyGreedySelectionOrder(mlir::Operation *anchor,
                           const mlir::egraph::EGraphExtractInfo &selection,
                           llvm::ArrayRef<llvm::StringRef> expectedSymbols) {
  return verifyExtractSelectionOrder(anchor, selection, expectedSymbols,
                                     "greedy");
}

mlir::LogicalResult verifyGreedyAliasSelection(
    mlir::Operation *anchor, const mlir::egraph::EGraphExtractInfo &selection,
    llvm::ArrayRef<llvm::StringRef> expectedAliasSymbols,
    llvm::ArrayRef<llvm::StringRef> expectedAliasTargets) {
  return verifyExtractAliasSelection(anchor, selection, expectedAliasSymbols,
                                     expectedAliasTargets, "greedy");
}

mlir::LogicalResult verifyGreedySelectedEClass(
    mlir::Operation *anchor, const mlir::egraph::EGraphExtractInfo &selection,
    llvm::StringRef symbolName, llvm::StringRef operationName,
    mlir::egraph::EGraphExtractCost expectedLocalCost,
    mlir::egraph::EGraphExtractCost expectedSubtreeCost) {
  return verifyExtractSelectedEClass(anchor, selection, symbolName,
                                     operationName, expectedLocalCost,
                                     expectedSubtreeCost, "greedy");
}

mlir::LogicalResult verifyLinearProgrammingSelectionOrder(
    mlir::Operation *anchor, const mlir::egraph::EGraphExtractInfo &selection,
    llvm::ArrayRef<llvm::StringRef> expectedSymbols);

mlir::LogicalResult verifyLinearProgrammingAliasSelection(
    mlir::Operation *anchor, const mlir::egraph::EGraphExtractInfo &selection,
    llvm::ArrayRef<llvm::StringRef> expectedAliasSymbols,
    llvm::ArrayRef<llvm::StringRef> expectedAliasTargets);

mlir::LogicalResult verifyLinearProgrammingSelectedEClass(
    mlir::Operation *anchor, const mlir::egraph::EGraphExtractInfo &selection,
    llvm::StringRef symbolName, llvm::StringRef operationName,
    mlir::egraph::EGraphExtractCost expectedLocalCost,
    mlir::egraph::EGraphExtractCost expectedSubtreeCost);

mlir::LogicalResult verifyGreedyExtract(mlir::ModuleOp module,
                                        mlir::egraph::EGraph &graph) {
  mlir::FailureOr<mlir::egraph::EGraphOp> egraph =
      getSingleActiveEGraph(module);
  if (mlir::failed(egraph))
    return module.emitError("expected one active egraph.egraph");

  if (egraph->getSymName() == "rebuilt_input_alias") {
    auto input = egraph->lookupSymbol<mlir::egraph::InputOp>("x");
    auto alias = egraph->lookupSymbol<mlir::egraph::EClassOp>("alias");
    if (!input || !alias)
      return egraph->emitOpError("expected @x and @alias symbols");

    graph.clearTouchedEClasses();
    if (mlir::failed(graph.unionValues(graph.getValue(input.getSymNameAttr()),
                                       graph.getValue(alias.getSymNameAttr()))))
      return alias.emitOpError("failed to seed rebuilt input alias test");

    if (mlir::failed(graph.rebuild(egraph->getOperation())))
      return egraph->emitOpError("rebuilt input alias graph failed to rebuild");

    egraph = module.lookupSymbol<mlir::egraph::EGraphOp>("rebuilt_input_alias");
    if (mlir::failed(egraph))
      return module.emitError("rebuilt input alias egraph disappeared");
    input = egraph->lookupSymbol<mlir::egraph::InputOp>("x");
    alias = egraph->lookupSymbol<mlir::egraph::EClassOp>("alias");
    if (!input || !alias)
      return egraph->emitOpError("rebuilt input alias graph lost @x or @alias");

    auto costModel = [](mlir::Operation *candidate)
        -> mlir::FailureOr<mlir::egraph::EGraphExtractCost> {
      if (candidate->getName().getStringRef() == "arith.addi")
        return mlir::egraph::EGraphExtractCost(1);
      return mlir::failure();
    };

    mlir::egraph::EGraphExtractInfo selection;
    if (mlir::failed(mlir::egraph::extractEGraphForTesting(
            graph, *egraph, mlir::egraph::EGraphExtractMode::Greedy, costModel,
            &selection)))
      return egraph->emitOpError(
          "failed to greedy extract rebuilt input alias graph");

    if (selection.roots.size() != 1 || selection.rootCosts.size() != 1 ||
        selection.selectedAliasEClasses.size() != 1 ||
        selection.selectedAliasTargets.size() != 1 ||
        !selection.selectedCandidateRoots.empty() ||
        !selection.selectedCandidateCosts.empty() ||
        !selection.selectedSubtreeCosts.empty())
      return egraph->emitOpError("unexpected rebuilt alias extract shape");

    if (selection.selectedAliasEClasses.front().getSymbolName() != "alias" ||
        selection.selectedAliasTargets.front().getSymbolName() != "x" ||
        selection.rootCosts.front() != 0)
      return egraph->emitOpError(
          "rebuilt input alias was not preserved by greedy extract");

    egraph->emitRemark()
        << "rebuilt input alias greedy extract preserved @"
        << selection.selectedAliasEClasses.front().getSymbolName() << " -> @"
        << selection.selectedAliasTargets.front().getSymbolName()
        << " cost=" << selection.rootCosts.front();
    return mlir::success();
  }

  mlir::Builder builder(module.getContext());
  llvm::SmallVector<mlir::StringAttr, 3> expectedRoots = {
      builder.getStringAttr("root"), builder.getStringAttr("loop"),
      builder.getStringAttr("alias_root")};

  auto costModel = [](mlir::Operation *candidate)
      -> mlir::FailureOr<mlir::egraph::EGraphExtractCost> {
    if (candidate->getName().getStringRef() == "arith.addi")
      return mlir::egraph::EGraphExtractCost(1);
    if (candidate->getName().getStringRef() == "arith.muli")
      return mlir::egraph::EGraphExtractCost(4);
    if (candidate->getName().getStringRef() == "arith.subi")
      return mlir::egraph::EGraphExtractCost(10);
    return mlir::failure();
  };

  mlir::egraph::EGraphExtractInfo selection;
  if (mlir::failed(mlir::egraph::extractEGraphForTesting(
          graph, *egraph, mlir::egraph::EGraphExtractMode::Greedy, costModel,
          &selection)))
    return egraph->emitOpError("failed to run greedy extract");

  if (selection.mode != mlir::egraph::EGraphExtractMode::Greedy ||
      selection.selectedEClasses.size() !=
          selection.selectedCandidateRoots.size() ||
      selection.selectedEClasses.size() !=
          selection.selectedCandidateCosts.size() ||
      selection.selectedEClasses.size() !=
          selection.selectedSubtreeCosts.size() ||
          selection.selectedAliasEClasses.size() !=
          selection.selectedAliasTargets.size())
    return egraph->emitOpError("unexpected greedy extract result shape");

  if (mlir::failed(verifyExtractRoots(egraph->getOperation(), selection.roots,
                                      expectedRoots)))
    return mlir::failure();

  if (selection.selectedEClasses.size() != 3)
    return egraph->emitOpError("expected three selected greedy candidates");

  llvm::SmallVector<llvm::StringRef, 3> expectedSelectedSymbols = {
      "cheap", "root", "loop"};
  llvm::SmallVector<llvm::StringRef, 2> expectedAliasSymbols = {"alias_leaf",
                                                                "alias_root"};
  llvm::SmallVector<llvm::StringRef, 2> expectedAliasTargets = {"cheap",
                                                                "alias_leaf"};
  if (mlir::failed(verifyGreedySelectionOrder(*egraph, selection,
                                              expectedSelectedSymbols)) ||
      mlir::failed(verifyGreedyAliasSelection(
          *egraph, selection, expectedAliasSymbols, expectedAliasTargets)) ||
      mlir::failed(
          verifyGreedySelectedEClass(*egraph, selection, "cheap", "arith.addi",
                                     mlir::egraph::EGraphExtractCost(1),
                                     mlir::egraph::EGraphExtractCost(1))) ||
      mlir::failed(
          verifyGreedySelectedEClass(*egraph, selection, "root", "arith.muli",
                                     mlir::egraph::EGraphExtractCost(4),
                                     mlir::egraph::EGraphExtractCost(5))) ||
      mlir::failed(
          verifyGreedySelectedEClass(*egraph, selection, "loop", "arith.muli",
                                     mlir::egraph::EGraphExtractCost(4),
                                     mlir::egraph::EGraphExtractCost(4))) ||
      selection.rootCosts.size() != 3 || selection.rootCosts[0] != 5 ||
      selection.rootCosts[1] != 4 || selection.rootCosts[2] != 1)
    return mlir::failure();

  if (mlir::succeeded(findExtractSelection(selection, "expensive")) ||
      mlir::succeeded(findExtractSelection(selection, "only_cycle")))
    return egraph->emitOpError(
        "greedy extract selected an unreachable or cyclic eclass");

  auto onlyCycle = egraph->lookupSymbol<mlir::egraph::EClassOp>("only_cycle");
  auto loop = egraph->lookupSymbol<mlir::egraph::EClassOp>("loop");
  auto aliasRoot = egraph->lookupSymbol<mlir::egraph::EClassOp>("alias_root");
  if (!onlyCycle || !loop || !aliasRoot)
    return egraph->emitOpError(
        "expected @only_cycle, @loop, and @alias_root symbols");

  llvm::SmallVector<mlir::egraph::EValue, 3> cyclicRoots = {
      graph.getValue(onlyCycle.getSymNameAttr()),
      graph.getValue(loop.getSymNameAttr()),
      graph.getValue(aliasRoot.getSymNameAttr())};
  mlir::egraph::EGraphExtractInfo cyclicSelection;
  if (mlir::succeeded(mlir::egraph::extractEGraphForTesting(
          graph, *egraph, mlir::egraph::EGraphExtractMode::Greedy, costModel,
          &cyclicSelection, cyclicRoots)))
    return onlyCycle.emitOpError(
        "greedy extract unexpectedly accepted a cyclic-only root");

  onlyCycle.emitRemark() << "greedy extract rejected cyclic-only root";
  return mlir::success();
}

mlir::LogicalResult
verifyLinearProgrammingExtract(mlir::ModuleOp module,
                               mlir::egraph::EGraph &graph) {
  mlir::FailureOr<mlir::egraph::EGraphOp> egraph =
      getSingleActiveEGraph(module);
  if (mlir::failed(egraph))
    return module.emitError("expected one active egraph.egraph");

  auto root1 = egraph->lookupSymbol<mlir::egraph::EClassOp>("root1");
  auto root2 = egraph->lookupSymbol<mlir::egraph::EClassOp>("root2");
  auto tieRoot = egraph->lookupSymbol<mlir::egraph::EClassOp>("tie_root");
  auto inputAliasRoot =
      egraph->lookupSymbol<mlir::egraph::EClassOp>("input_alias_root");
  auto cycle = egraph->lookupSymbol<mlir::egraph::EClassOp>("cycle");
  if (!root1 || !root2 || !tieRoot || !inputAliasRoot || !cycle)
    return egraph->emitOpError("expected @root1, @root2, @tie_root, "
                               "@input_alias_root, and @cycle symbols");

  llvm::SmallVector<mlir::StringAttr, 4> expectedRoots = {
      root1.getSymNameAttr(), root2.getSymNameAttr(), tieRoot.getSymNameAttr(),
      inputAliasRoot.getSymNameAttr()};

  auto costModel = [](mlir::Operation *candidate)
      -> mlir::FailureOr<mlir::egraph::EGraphExtractCost> {
    auto ownerOp = llvm::dyn_cast_or_null<mlir::egraph::EClassOp>(
        candidate->getBlock()->getParentOp());
    if (!ownerOp)
      return mlir::failure();
    llvm::StringRef owner = ownerOp.getSymName();
    llvm::StringRef operationName = candidate->getName().getStringRef();
    if (owner == "shared" && operationName == "arith.constant")
      return mlir::egraph::EGraphExtractCost(5);
    if ((owner == "a" || owner == "b") && operationName == "arith.constant")
      return mlir::egraph::EGraphExtractCost(1);
    if (owner == "root1" && operationName == "arith.addi")
      return mlir::egraph::EGraphExtractCost(1);
    if (owner == "root1" && operationName == "arith.muli")
      return mlir::egraph::EGraphExtractCost(2);
    if (owner == "root2" && operationName == "arith.addi")
      return mlir::egraph::EGraphExtractCost(1);
    if (owner == "tie_root" && operationName == "arith.addi")
      return mlir::egraph::EGraphExtractCost(1);
    if (owner == "tie_root" && operationName == "arith.muli")
      return mlir::egraph::EGraphExtractCost(1);
    if (owner == "cycle" && operationName == "arith.addi")
      return mlir::egraph::EGraphExtractCost(1);
    return mlir::failure();
  };

  mlir::egraph::EGraphExtractInfo selection;
  if (mlir::failed(mlir::egraph::extractEGraphForTesting(
          graph, *egraph,
          mlir::egraph::EGraphExtractMode::LinearProgramming, costModel,
          &selection)))
    return egraph->emitOpError("failed to run LP extract");

  if (selection.mode != mlir::egraph::EGraphExtractMode::LinearProgramming ||
      selection.selectedEClasses.size() !=
          selection.selectedCandidateRoots.size() ||
      selection.selectedEClasses.size() !=
          selection.selectedCandidateCosts.size() ||
      selection.selectedEClasses.size() !=
          selection.selectedSubtreeCosts.size() ||
      selection.selectedAliasEClasses.size() !=
          selection.selectedAliasTargets.size())
    return egraph->emitOpError("unexpected LP extract result shape");

  if (mlir::failed(verifyExtractRoots(egraph->getOperation(), selection.roots,
                                      expectedRoots)))
    return mlir::failure();

  llvm::SmallVector<llvm::StringRef, 4> expectedSelectedSymbols = {
      "shared", "root1", "root2", "tie_root"};
  llvm::SmallVector<llvm::StringRef, 2> expectedAliasSymbols = {
      "input_alias_leaf", "input_alias_root"};
  llvm::SmallVector<llvm::StringRef, 2> expectedAliasTargets = {
      "x", "input_alias_leaf"};
  if (mlir::failed(verifyLinearProgrammingSelectionOrder(
          *egraph, selection, expectedSelectedSymbols)) ||
      mlir::failed(verifyLinearProgrammingAliasSelection(
          *egraph, selection, expectedAliasSymbols, expectedAliasTargets)) ||
      mlir::failed(verifyLinearProgrammingSelectedEClass(
          *egraph, selection, "shared", "arith.constant",
          mlir::egraph::EGraphExtractCost(5),
          mlir::egraph::EGraphExtractCost(5))) ||
      mlir::failed(verifyLinearProgrammingSelectedEClass(
          *egraph, selection, "root1", "arith.muli",
          mlir::egraph::EGraphExtractCost(2),
          mlir::egraph::EGraphExtractCost(7))) ||
      mlir::failed(verifyLinearProgrammingSelectedEClass(
          *egraph, selection, "root2", "arith.addi",
          mlir::egraph::EGraphExtractCost(1),
          mlir::egraph::EGraphExtractCost(6))) ||
      mlir::failed(verifyLinearProgrammingSelectedEClass(
          *egraph, selection, "tie_root", "arith.addi",
          mlir::egraph::EGraphExtractCost(1),
          mlir::egraph::EGraphExtractCost(1))) ||
      selection.rootCosts.size() != 4 || selection.rootCosts[0] != 7 ||
      selection.rootCosts[1] != 6 || selection.rootCosts[2] != 1 ||
      selection.rootCosts[3] != 0)
    return mlir::failure();

  if (mlir::succeeded(findExtractSelection(selection, "a")) ||
      mlir::succeeded(findExtractSelection(selection, "b")))
    return egraph->emitOpError(
        "LP extract selected a branch that is not globally optimal");

  mlir::egraph::EGraphExtractCost globalCost = 0;
  for (mlir::egraph::EGraphExtractCost cost : selection.selectedCandidateCosts)
    globalCost += cost;
  if (globalCost != 9)
    return egraph->emitOpError("LP extract selected global cost ")
           << globalCost << ", expected 9";

  egraph->emitRemark() << "lp extract root costs -> @root1="
                       << selection.rootCosts[0]
                       << ", @root2=" << selection.rootCosts[1]
                       << ", @tie_root=" << selection.rootCosts[2]
                       << ", @input_alias_root=" << selection.rootCosts[3]
                       << " global_cost=" << globalCost;

  llvm::SmallVector<mlir::egraph::EValue, 4> cyclicRoots = {
      graph.getValue(cycle.getSymNameAttr()),
      graph.getValue(root2.getSymNameAttr()),
      graph.getValue(tieRoot.getSymNameAttr()),
      graph.getValue(inputAliasRoot.getSymNameAttr())};
  mlir::egraph::EGraphExtractInfo cyclicSelection;
  if (mlir::succeeded(mlir::egraph::extractEGraphForTesting(
          graph, *egraph,
          mlir::egraph::EGraphExtractMode::LinearProgramming, costModel,
          &cyclicSelection, cyclicRoots)))
    return cycle.emitOpError("LP extract unexpectedly accepted a cyclic root");

  cycle.emitRemark() << "lp extract rejected cyclic root";
  return mlir::success();
}

mlir::LogicalResult verifyLinearProgrammingSelectionOrder(
    mlir::Operation *anchor, const mlir::egraph::EGraphExtractInfo &selection,
    llvm::ArrayRef<llvm::StringRef> expectedSymbols) {
  return verifyExtractSelectionOrder(anchor, selection, expectedSymbols, "lp");
}

mlir::LogicalResult verifyLinearProgrammingAliasSelection(
    mlir::Operation *anchor, const mlir::egraph::EGraphExtractInfo &selection,
    llvm::ArrayRef<llvm::StringRef> expectedAliasSymbols,
    llvm::ArrayRef<llvm::StringRef> expectedAliasTargets) {
  return verifyExtractAliasSelection(anchor, selection, expectedAliasSymbols,
                                     expectedAliasTargets, "lp");
}

mlir::LogicalResult verifyLinearProgrammingSelectedEClass(
    mlir::Operation *anchor, const mlir::egraph::EGraphExtractInfo &selection,
    llvm::StringRef symbolName, llvm::StringRef operationName,
    mlir::egraph::EGraphExtractCost expectedLocalCost,
    mlir::egraph::EGraphExtractCost expectedSubtreeCost) {
  return verifyExtractSelectedEClass(anchor, selection, symbolName,
                                     operationName, expectedLocalCost,
                                     expectedSubtreeCost, "lp");
}

mlir::LogicalResult
verifyGreedyExtractMaterialization(mlir::ModuleOp module,
                                   mlir::egraph::EGraph &graph) {
  mlir::FailureOr<mlir::egraph::EGraphOp> egraph =
      getSingleActiveEGraph(module);
  if (mlir::failed(egraph))
    return module.emitError("expected one active egraph.egraph");

  auto costModel = [](mlir::Operation *candidate)
      -> mlir::FailureOr<mlir::egraph::EGraphExtractCost> {
    llvm::StringRef operationName = candidate->getName().getStringRef();
    if (operationName == "arith.addi" || operationName == "arith.muli" ||
        operationName == "arith.subi")
      return mlir::egraph::EGraphExtractCost(1);
    return mlir::failure();
  };

  mlir::egraph::EGraphExtractInfo selection;
  if (mlir::failed(mlir::egraph::extractEGraphForTesting(
          graph, *egraph, mlir::egraph::EGraphExtractMode::Greedy, costModel,
          &selection)))
    return egraph->emitOpError(
        "failed to run greedy extract for materialization");

  llvm::StringRef egraphName = egraph->getSymName();
  if (egraphName.ends_with("_egraph"))
    egraphName = egraphName.drop_back(7);
  std::string materializedName = (egraphName + "_materialized").str();
  if (module.lookupSymbol<mlir::func::FuncOp>(materializedName))
    return module.emitError("expected materialized function name to be unused");

  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointAfter(egraph->getOperation());
  auto funcType = builder.getFunctionType(egraph->getArgumentTypes(),
                                          egraph->getResultTypes());
  mlir::func::FuncOp materializedFunc = mlir::func::FuncOp::create(
      builder, egraph->getLoc(), materializedName, funcType);
  mlir::Block *entryBlock = materializedFunc.addEntryBlock();
  builder.setInsertionPointToEnd(entryBlock);

  llvm::SmallVector<mlir::Value, 4> inputValues(
      entryBlock->getArguments().begin(), entryBlock->getArguments().end());
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> roots =
      mlir::egraph::materializeEGraphExtractInfoForTesting(
          graph, *egraph, selection, builder, inputValues);
  if (mlir::failed(roots))
    return egraph->emitOpError("failed to materialize extracted roots");

  if (roots->size() != selection.roots.size())
    return egraph->emitOpError("materialized root count diverged");

  mlir::func::ReturnOp::create(builder, egraph->getLoc(), *roots);
  module.emitRemark() << "egraph extract materialized @" << egraph->getSymName()
                      << " into @" << materializedFunc.getSymName();
  return mlir::success();
}

mlir::LogicalResult
verifySymbolicEValueLookupGraph(mlir::egraph::EGraphOp egraph) {
  auto input = egraph.lookupSymbol<mlir::egraph::InputOp>("x");
  auto constant = egraph.lookupSymbol<mlir::egraph::EClassOp>("c2");
  auto doubled = egraph.lookupSymbol<mlir::egraph::EClassOp>("double");
  if (!input || !constant || !doubled)
    return egraph.emitOpError("expected @x, @c2, and @double symbols");

  mlir::egraph::EGraph graph;
  if (mlir::failed(graph.indexEGraph(egraph)))
    return egraph.emitOpError("failed to index symbolic egraph");

  auto inputRef = mlir::FlatSymbolRefAttr::get(input.getSymNameAttr());
  mlir::egraph::EValue inputValue = graph.getValue(inputRef);
  if (inputValue.getResultIndex() != 0 ||
      inputValue.getType() != input.getPayloadType())
    return input.emitOpError("constructed input EValue had wrong payload type");
  input.emitRemark() << "constructed input EValue @" << input.getSymName()
                     << " result #" << inputValue.getResultIndex() << " : "
                     << inputValue.getType();

  mlir::FailureOr<mlir::egraph::EValue> lookedUpInput =
      graph.lookupValue(input.getValue());
  if (mlir::failed(lookedUpInput) ||
      lookedUpInput->getSymbolRef() != inputRef ||
      lookedUpInput->getResultIndex() != 0)
    return input.emitOpError("lookupValue(entry arg) did not resolve to @x");
  input.emitRemark() << "lookupValue(entry arg) -> @"
                     << lookedUpInput->getSymbolRef().getValue();

  mlir::egraph::EValue unsupportedSlot = graph.getValue(inputRef, 1);
  if (unsupportedSlot.getType() || !unsupportedSlot.getDefs().empty())
    return input.emitOpError(
        "symbolic nonzero result slot should stay inactive in v1.1");
  input.emitRemark() << "symbolic EValue result slot #"
                     << unsupportedSlot.getResultIndex()
                     << " is reserved for future multi-result support";

  auto constantRef = mlir::FlatSymbolRefAttr::get(constant.getSymNameAttr());
  mlir::egraph::EValue constantValue = graph.getValue(constantRef);
  if (constantValue.getResultIndex() != 0 ||
      constantValue.getType() != constant.getPayloadType())
    return constant.emitOpError(
        "constructed eclass EValue had wrong payload type");
  constant.emitRemark() << "constructed eclass EValue @"
                        << constant.getSymName() << " result #"
                        << constantValue.getResultIndex() << " : "
                        << constantValue.getType();

  mlir::Block &doubleBlock = doubled.getCandidates().front().front();
  if (doubleBlock.getNumArguments() != 2)
    return doubled.emitOpError("expected @double candidate with two arguments");

  mlir::FailureOr<mlir::egraph::EValue> lhsValue =
      graph.lookupValue(doubleBlock.getArgument(0));
  mlir::FailureOr<mlir::egraph::EValue> rhsValue =
      graph.lookupValue(doubleBlock.getArgument(1));
  if (mlir::failed(lhsValue) || lhsValue->getSymbolRef() != inputRef)
    return doubled.emitOpError("candidate argument #0 did not resolve to @x");
  if (mlir::failed(rhsValue) || rhsValue->getSymbolRef() != constantRef)
    return doubled.emitOpError("candidate argument #1 did not resolve to @c2");
  doubled.emitRemark() << "lookupValue(candidate arg #0) -> @"
                       << lhsValue->getSymbolRef().getValue();
  doubled.emitRemark() << "lookupValue(candidate arg #1) -> @"
                       << rhsValue->getSymbolRef().getValue();

  auto yield =
      mlir::dyn_cast<mlir::egraph::YieldOp>(doubleBlock.getTerminator());
  if (!yield)
    return doubled.emitOpError("expected @double candidate terminator");

  auto yieldedResult = mlir::dyn_cast<mlir::OpResult>(yield.getOperand(0));
  if (!yieldedResult)
    return doubled.emitOpError("expected @double to yield an op result");

  auto doubledRef = mlir::FlatSymbolRefAttr::get(doubled.getSymNameAttr());
  mlir::FailureOr<mlir::egraph::EValue> lookedUpYield =
      graph.lookupValue(yieldedResult);
  if (mlir::failed(lookedUpYield) ||
      lookedUpYield->getSymbolRef() != doubledRef)
    return doubled.emitOpError(
        "yielded payload value did not resolve to @double");
  doubled.emitRemark() << "lookupValue(yielded payload) -> @"
                       << lookedUpYield->getSymbolRef().getValue();

  auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(yieldedResult.getOwner());
  if (!mul)
    return doubled.emitOpError(
        "expected @double candidate root to be arith.muli");

  mlir::FailureOr<mlir::egraph::EOpRefBase> lookedUpRef =
      graph.lookupOpRef(mul.getOperation());
  if (mlir::failed(lookedUpRef))
    return doubled.emitOpError("lookupOpRef(candidate root) failed");

  mlir::egraph::EOpRef<mlir::arith::MulIOp> typedRef(*lookedUpRef);
  if (typedRef.getOperation() != mul.getOperation())
    return doubled.emitOpError("typed EOpRef resolved the wrong root op");
  if (typedRef.getOperand(0).getSymbolRef() != inputRef ||
      typedRef.getOperand(1).getSymbolRef() != constantRef)
    return doubled.emitOpError(
        "EOpRef operands did not resolve to child symbols");
  if (!typedRef.hasResult(0) ||
      typedRef.getResult(0).getSymbolRef() != doubledRef ||
      typedRef.getResult(0).getResultIndex() != 0)
    return doubled.emitOpError("EOpRef result did not resolve to owner eclass");

  doubled.emitRemark() << "lookupOpRef(candidate root) -> "
                       << typedRef.getOperationName();
  doubled.emitRemark() << "EOpRef operand #0 -> @"
                       << typedRef.getOperand(0).getSymbolRef().getValue();
  doubled.emitRemark() << "EOpRef operand #1 -> @"
                       << typedRef.getOperand(1).getSymbolRef().getValue();
  doubled.emitRemark() << "EOpRef result #0 -> @"
                       << typedRef.getResult(0).getSymbolRef().getValue()
                       << " result #" << typedRef.getResult(0).getResultIndex();

  return mlir::success();
}

mlir::LogicalResult
verifySymbolicEValueQueryGraph(mlir::egraph::EGraphOp egraph) {
  auto inputX = egraph.lookupSymbol<mlir::egraph::InputOp>("x");
  auto inputY = egraph.lookupSymbol<mlir::egraph::InputOp>("y");
  auto lhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("lhs");
  auto rhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("rhs");
  if (!inputX || !inputY || !lhs || !rhs)
    return egraph.emitOpError("expected @x, @y, @lhs, and @rhs symbols");

  mlir::egraph::EGraph graph;
  if (mlir::failed(graph.indexEGraph(egraph)))
    return egraph.emitOpError("failed to index symbolic query egraph");

  mlir::egraph::EValue inputValue = graph.getValue(inputX.getSymNameAttr());
  if (!inputValue.getDefs().empty())
    return inputX.emitOpError("input unexpectedly had candidate roots");
  inputX.emitRemark() << "EValue getDefs returned no candidate roots for @"
                      << inputX.getSymName();

  graph.clearTouchedEClasses();
  mlir::FailureOr<mlir::egraph::EGraphUnionResult> unioned =
      graph.unionValues(graph.getValue(lhs.getSymNameAttr()),
                        graph.getValue(rhs.getSymNameAttr()));
  if (mlir::failed(unioned) || !unioned->changed)
    return rhs.emitOpError("failed to seed symbolic query union");

  mlir::StringAttr lhsName = lhs.getSymNameAttr();
  mlir::StringAttr rhsName = rhs.getSymNameAttr();
  mlir::FailureOr<mlir::egraph::EGraphRebuildResult> rebuilt =
      graph.rebuild(egraph);
  if (mlir::failed(rebuilt))
    return egraph.emitOpError("symbolic query rebuild failed");

  lhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("lhs");
  rhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("rhs");
  if (!lhs)
    return egraph.emitOpError("symbolic query rebuild lost @lhs");
  if (rhs)
    return rhs.emitOpError(
        "symbolic query rebuild did not remove absorbed member eclass");

  mlir::egraph::EValue rhsValue = graph.getValue(rhsName);
  mlir::egraph::EValue leaderValue = rhsValue.getLeader();
  if (leaderValue.getSymbolNameAttr() != lhsName)
    return lhs.emitOpError("EValue query did not follow the current leader");
  lhs.emitRemark() << "EValue query followed leader @"
                   << leaderValue.getSymbolName();

  llvm::SmallVector<mlir::egraph::EOpRefBase> allDefs = rhsValue.getDefs();
  if (allDefs.size() != 3)
    return lhs.emitOpError("expected three candidate roots after union");
  for (mlir::egraph::EOpRefBase ref : allDefs) {
    if (!ref.isLive())
      return lhs.emitOpError("getDefs returned a stale candidate root");
    if (!ref.hasResult(0) || ref.getResult(0).getLeader() != leaderValue)
      return lhs.emitOpError("getDefs returned a root outside the leader");
  }
  lhs.emitRemark() << "EValue getDefs found " << allDefs.size()
                   << " candidate root(s) through the leader";

  llvm::SmallVector<mlir::egraph::EOpRef<mlir::egraph::test::OpAOp>> opADefs =
      rhsValue.getDefs<mlir::egraph::test::OpAOp>();
  if (opADefs.size() != 2)
    return lhs.emitOpError("expected two op_a definitions after union");
  lhs.emitRemark() << "EValue typed getDefs found " << opADefs.size()
                   << " op_a candidate(s)";

  if (!rhsValue.hasDef<mlir::egraph::test::OpAOp>() ||
      !rhsValue.hasDef<mlir::egraph::test::OpBOp>() ||
      rhsValue.hasDef<mlir::egraph::test::LeafOp>())
    return lhs.emitOpError("hasDef did not reflect merged symbolic defs");
  lhs.emitRemark() << "EValue hasDef observed op_a and op_b defs after union";

  mlir::FailureOr<mlir::egraph::EOpRef<mlir::egraph::test::OpBOp>> uniqueOpB =
      rhsValue.getUniqueDef<mlir::egraph::test::OpBOp>();
  if (mlir::failed(uniqueOpB))
    return lhs.emitOpError("expected a unique op_b definition after union");
  (*uniqueOpB).getOperation()->emitRemark()
      << "EValue getUniqueDef returned the unique op_b candidate";

  if (mlir::succeeded(rhsValue.getUniqueDef<mlir::egraph::test::OpAOp>()))
    return lhs.emitOpError("op_a unexpectedly had a unique definition");
  lhs.emitRemark() << "EValue getUniqueDef rejected multiple op_a candidates";

  bool matchedSlowByLogicalResult = false;
  if (mlir::failed(rhsValue.matchDef<mlir::egraph::test::OpAOp>(
          [&](mlir::egraph::EOpRef<mlir::egraph::test::OpAOp> opA) {
            auto modeAttr =
                opA.getOperation()->getAttrOfType<mlir::StringAttr>("mode");
            if (!modeAttr || modeAttr.getValue() != "slow")
              return mlir::failure();
            matchedSlowByLogicalResult = true;
            opA.getOperation()->emitRemark()
                << "EValue matchDef accepted LogicalResult callback";
            return mlir::success();
          })))
    return lhs.emitOpError("LogicalResult matchDef did not match op_a");
  if (!matchedSlowByLogicalResult)
    return lhs.emitOpError("LogicalResult matchDef did not run callback");

  bool matchedSlowByBool = false;
  if (mlir::failed(rhsValue.matchDef<mlir::egraph::test::OpAOp>(
          [&](mlir::egraph::EOpRef<mlir::egraph::test::OpAOp> opA) {
            auto modeAttr =
                opA.getOperation()->getAttrOfType<mlir::StringAttr>("mode");
            bool matched = modeAttr && modeAttr.getValue() == "slow";
            if (matched) {
              matchedSlowByBool = true;
              opA.getOperation()->emitRemark()
                  << "EValue matchDef accepted bool callback";
            }
            return matched;
          })))
    return lhs.emitOpError("bool matchDef did not match op_a");
  if (!matchedSlowByBool)
    return lhs.emitOpError("bool matchDef did not run callback");

  if (mlir::succeeded(rhsValue.matchDef<mlir::egraph::test::OpAOp>(
          [&](mlir::egraph::EOpRef<mlir::egraph::test::OpAOp> opA) {
            (void)opA;
            return false;
          })))
    return lhs.emitOpError("bool matchDef accepted an all-failing predicate");
  lhs.emitRemark() << "EValue matchDef rejected all-false bool callback";

  unsigned voidVisits = 0;
  bool sawSlowInVoidTraversal = false;
  (void)rhsValue.matchDef<mlir::egraph::test::OpAOp>(
      [&](mlir::egraph::EOpRef<mlir::egraph::test::OpAOp> opA) {
        ++voidVisits;
        auto modeAttr =
            opA.getOperation()->getAttrOfType<mlir::StringAttr>("mode");
        sawSlowInVoidTraversal |= modeAttr && modeAttr.getValue() == "slow";
      });
  if (voidVisits != 2 || !sawSlowInVoidTraversal)
    return lhs.emitOpError("void matchDef did not visit every op_a candidate");
  lhs.emitRemark() << "EValue matchDef visited " << voidVisits
                   << " op_a candidate(s) with a void callback";

  return mlir::success();
}

mlir::LogicalResult verifySymbolicEValueLookup(mlir::ModuleOp module) {
  if (auto egraph =
          module.lookupSymbol<mlir::egraph::EGraphOp>("symbolic_lookup"))
    return verifySymbolicEValueLookupGraph(egraph);
  if (auto egraph =
          module.lookupSymbol<mlir::egraph::EGraphOp>("symbolic_query"))
    return verifySymbolicEValueQueryGraph(egraph);
  return module.emitError(
      "expected egraph.egraph @symbolic_lookup or @symbolic_query");
}

mlir::FailureOr<mlir::egraph::EOpRefBase>
getOnlyCandidateRoot(mlir::egraph::EGraph &graph,
                     mlir::egraph::EClassOp eclass) {
  llvm::SmallVector<mlir::egraph::EOpRefBase> refs =
      graph.getCandidateRoots(eclass);
  if (refs.size() != 1)
    return mlir::failure();
  return refs.front();
}

mlir::LogicalResult verifySymbolicIndex(mlir::ModuleOp module) {
  auto egraph = module.lookupSymbol<mlir::egraph::EGraphOp>("symbolic_index");
  if (!egraph)
    return module.emitError("expected egraph.egraph @symbolic_index");

  auto inputX = egraph.lookupSymbol<mlir::egraph::InputOp>("x");
  auto inputY = egraph.lookupSymbol<mlir::egraph::InputOp>("y");
  auto constant = egraph.lookupSymbol<mlir::egraph::EClassOp>("c2");
  auto mulA = egraph.lookupSymbol<mlir::egraph::EClassOp>("mul_a");
  auto mulB = egraph.lookupSymbol<mlir::egraph::EClassOp>("mul_b");
  auto mulY = egraph.lookupSymbol<mlir::egraph::EClassOp>("mul_y");
  if (!inputX || !inputY || !constant || !mulA || !mulB || !mulY)
    return module.emitError(
        "expected @x, @y, @c2, @mul_a, @mul_b, and @mul_y symbols");

  mlir::egraph::EGraph graph;
  if (mlir::failed(graph.indexEGraph(egraph)))
    return module.emitError("failed to index symbolic egraph");

  mlir::egraph::EValue xValue = graph.getValue(inputX.getSymNameAttr());
  llvm::SmallVector<mlir::egraph::EOpRefBase> parentCandidates =
      graph.getParentCandidates(xValue);
  if (parentCandidates.size() != 2)
    return inputX.emitOpError("expected two parent candidates for @x");
  for (mlir::egraph::EOpRefBase ref : parentCandidates) {
    if (!ref.isLive() || !llvm::isa<mlir::arith::MulIOp>(ref.getOperation()))
      return inputX.emitOpError("parent index for @x returned a wrong root");
  }
  inputX.emitRemark() << "parent index for @x: " << parentCandidates.size()
                      << " candidate(s)";

  mlir::FailureOr<mlir::egraph::EOpRefBase> mulARef =
      getOnlyCandidateRoot(graph, mulA);
  mlir::FailureOr<mlir::egraph::EOpRefBase> mulBRef =
      getOnlyCandidateRoot(graph, mulB);
  mlir::FailureOr<mlir::egraph::EOpRefBase> mulYRef =
      getOnlyCandidateRoot(graph, mulY);
  if (mlir::failed(mulARef) || mlir::failed(mulBRef) || mlir::failed(mulYRef))
    return egraph.emitOpError("expected one root op per symbolic eclass");

  mlir::FailureOr<mlir::egraph::EGraphStructuralKey> keyA =
      graph.getStructuralKey(*mulARef);
  mlir::FailureOr<mlir::egraph::EGraphStructuralKey> keyB =
      graph.getStructuralKey(*mulBRef);
  mlir::FailureOr<mlir::egraph::EGraphStructuralKey> keyY =
      graph.getStructuralKey(*mulYRef);
  if (mlir::failed(keyA) || mlir::failed(keyB) || mlir::failed(keyY))
    return egraph.emitOpError("expected structural keys for symbolic roots");

  if (!keyA->childKeyValues.empty() || keyA->childLeaderSymbols.size() != 2 ||
      keyA->childLeaderSymbols[0] != inputX.getSymNameAttr() ||
      keyA->childLeaderSymbols[1] != constant.getSymNameAttr())
    return mulA.emitOpError(
        "structural key did not record @mul_a child leaders");

  if (*keyA != *keyB)
    return mulB.emitOpError("duplicate symbolic candidates did not share key");

  mlir::FailureOr<mlir::egraph::EGraphHashConsEntry> duplicateEntry =
      graph.lookupStructuralEntry(*mulBRef);
  if (mlir::failed(duplicateEntry))
    return mulB.emitOpError("duplicate symbolic structural lookup failed");
  if (duplicateEntry->operation != mulARef->getOperation() ||
      duplicateEntry->eclass != mulA ||
      duplicateEntry->resultSymbols.size() != 1 ||
      duplicateEntry->resultSymbols.front() != mulA.getSymNameAttr())
    return mulB.emitOpError("duplicate symbolic key did not reuse @mul_a");
  mulB.emitRemark() << "structural hashcons reused @mul_a";

  if (!keyY->childKeyValues.empty() || keyY->childLeaderSymbols.size() != 2 ||
      keyY->childLeaderSymbols[0] != inputY.getSymNameAttr() ||
      keyY->childLeaderSymbols[1] != constant.getSymNameAttr() ||
      *keyY == *keyA)
    return mulY.emitOpError(
        "distinct child leaders did not keep symbolic structural key unique");

  mlir::FailureOr<mlir::egraph::EGraphHashConsEntry> distinctEntry =
      graph.lookupStructuralEntry(*mulYRef);
  if (mlir::failed(distinctEntry) ||
      distinctEntry->operation != mulYRef->getOperation() ||
      distinctEntry->eclass != mulY)
    return mulY.emitOpError(
        "distinct symbolic structural key reused an unrelated occurrence");
  mulY.emitRemark() << "structural key kept different child symbols separate";

  graph.clearIndex();
  if (mlir::failed(graph.indexEGraph(egraph)))
    return module.emitError("failed to rebuild symbolic indexes");

  parentCandidates = graph.getParentCandidates(xValue);
  if (parentCandidates.size() != 2)
    return inputX.emitOpError("rebuilt parent index for @x was incomplete");
  inputX.emitRemark() << "rebuilt parent index for @x: "
                      << parentCandidates.size() << " candidate(s)";

  return mlir::success();
}

mlir::FailureOr<mlir::StringAttr>
getSingleCandidateRef(mlir::egraph::EClassOp eclass,
                      unsigned candidateOrdinal) {
  mlir::ArrayAttr candidateRefs = eclass.getCandidateRefs();
  if (candidateOrdinal >= candidateRefs.size())
    return mlir::failure();

  auto row = mlir::dyn_cast<mlir::ArrayAttr>(candidateRefs[candidateOrdinal]);
  if (!row || row.size() != 1)
    return mlir::failure();

  auto symbolRef = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(row[0]);
  if (!symbolRef)
    return mlir::failure();
  return symbolRef.getAttr();
}

mlir::FailureOr<mlir::egraph::EOpRefBase>
getUniqueLiveRootForEClass(mlir::egraph::EGraph &graph,
                           mlir::egraph::EClassOp eclass) {
  mlir::egraph::EOpRefBase found;
  bool seen = false;
  for (mlir::egraph::EOpRefBase ref : graph.getOpRefs()) {
    if (ref.getEClassOp() != eclass)
      continue;
    if (seen)
      return mlir::failure();
    found = ref;
    seen = true;
  }

  if (!seen)
    return mlir::failure();
  return found;
}

mlir::LogicalResult verifySymbolicUnion(mlir::ModuleOp module) {
  auto egraph = module.lookupSymbol<mlir::egraph::EGraphOp>("symbolic_union");
  if (!egraph)
    return module.emitError("expected egraph.egraph @symbolic_union");

  auto lhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("lhs");
  auto rhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("rhs");
  auto user = egraph.lookupSymbol<mlir::egraph::EClassOp>("user");
  if (!lhs || !rhs || !user)
    return module.emitError("expected @lhs, @rhs, and @user symbols");

  auto returnOp = mlir::dyn_cast<mlir::egraph::ReturnOp>(
      egraph.getBody().front().getTerminator());
  if (!returnOp)
    return egraph.emitOpError("expected symbolic egraph terminator");

  mlir::egraph::EGraph graph;
  if (mlir::failed(graph.indexEGraph(egraph)))
    return module.emitError("failed to index symbolic union egraph");

  mlir::egraph::EValue lhsValue = graph.getValue(lhs.getSymNameAttr());
  mlir::egraph::EValue rhsValue = graph.getValue(rhs.getSymNameAttr());
  graph.clearTouchedEClasses();

  mlir::FailureOr<mlir::egraph::EGraphUnionResult> unioned =
      graph.unionValues(lhsValue, rhsValue);
  if (mlir::failed(unioned) || !unioned->changed)
    return rhs.emitOpError("failed to union symbolic eclasses");
  if (rhsValue.getLeader().getSymbolNameAttr() != lhs.getSymNameAttr())
    return rhs.emitOpError("symbolic union did not keep @lhs as leader");
  if (!llvm::is_contained(unioned->touchedEClasses, lhs) ||
      !llvm::is_contained(unioned->touchedEClasses, rhs))
    return rhs.emitOpError("symbolic union did not mark touched eclasses");
  rhs.emitRemark()
      << "symbolic union resolved @rhs to leader @lhs before rebuild";

  mlir::FailureOr<mlir::egraph::EGraphRebuildResult> rebuilt =
      graph.rebuild(egraph);
  if (mlir::failed(rebuilt))
    return module.emitError("symbolic union rebuild failed");
  lhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("lhs");
  rhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("rhs");
  user = egraph.lookupSymbol<mlir::egraph::EClassOp>("user");
  if (!lhs || !user)
    return module.emitError("symbolic union rebuild lost @lhs or @user");
  if (rhs)
    return rhs.emitOpError(
        "symbolic union rebuild did not remove absorbed member eclass");
  bool sawAffectedUser = false;
  for (mlir::egraph::EOpRefBase ref : rebuilt->affectedParentCandidates) {
    auto opB = llvm::dyn_cast<mlir::egraph::test::OpBOp>(ref.getOperation());
    if (!opB)
      continue;
    auto tagAttr = opB->getAttrOfType<mlir::StringAttr>("tag");
    sawAffectedUser |= tagAttr && tagAttr.getValue() == "user";
  }
  if (!sawAffectedUser)
    return user.emitOpError(
        "symbolic union rebuild lost affected parent candidate");
  user.emitRemark()
      << "symbolic union rebuild returned affected parent candidates";

  if (returnOp.getTargets().size() != 1 ||
      mlir::cast<mlir::FlatSymbolRefAttr>(returnOp.getTargets()[0]).getAttr() !=
          lhs.getSymNameAttr())
    return returnOp.emitOpError(
        "symbolic union rebuild did not retarget return");

  llvm::SmallVector<mlir::egraph::EOpRefBase> mergedRoots =
      graph.getCandidateRoots(lhsValue);
  bool sawLhsCandidate = false;
  bool sawRhsCandidate = false;
  for (mlir::egraph::EOpRefBase ref : mergedRoots) {
    if (auto opB =
            llvm::dyn_cast<mlir::egraph::test::OpBOp>(ref.getOperation())) {
      auto tagAttr = opB->getAttrOfType<mlir::StringAttr>("tag");
      sawLhsCandidate |= tagAttr && tagAttr.getValue() == "lhs";
      continue;
    }

    if (auto opA =
            llvm::dyn_cast<mlir::egraph::test::OpAOp>(ref.getOperation())) {
      auto modeAttr = opA->getAttrOfType<mlir::StringAttr>("mode");
      sawRhsCandidate |= modeAttr && modeAttr.getValue() == "rhs";
    }
  }
  if (!sawLhsCandidate || !sawRhsCandidate)
    return lhs.emitOpError(
        "symbolic leader did not expose merged candidate roots after rebuild");
  lhs.emitRemark()
      << "symbolic union leader defs included original rhs candidate";

  lhs.emitRemark() << "symbolic union rebuild removed absorbed rhs eclass";

  return mlir::success();
}

mlir::LogicalResult verifySymbolicFixedPoint(mlir::ModuleOp module) {
  auto egraph =
      module.lookupSymbol<mlir::egraph::EGraphOp>("symbolic_fixed_point");
  if (!egraph)
    return module.emitError("expected egraph.egraph @symbolic_fixed_point");

  auto inputX = egraph.lookupSymbol<mlir::egraph::InputOp>("x");
  auto lhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("lhs");
  auto rhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("rhs");
  auto parentLhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("parent_lhs");
  auto parentRhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("parent_rhs");
  auto grandLhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("grand_lhs");
  auto grandRhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("grand_rhs");
  auto probe = egraph.lookupSymbol<mlir::egraph::EClassOp>("probe");
  auto selfCycle = egraph.lookupSymbol<mlir::egraph::EClassOp>("self_cycle");
  auto mutualA = egraph.lookupSymbol<mlir::egraph::EClassOp>("mutual_a");
  auto mutualB = egraph.lookupSymbol<mlir::egraph::EClassOp>("mutual_b");
  if (!inputX || !lhs || !rhs || !parentLhs || !parentRhs || !grandLhs ||
      !grandRhs || !probe || !selfCycle || !mutualA || !mutualB)
    return module.emitError("expected symbolic fixed-point test eclasses");

  auto returnOp = mlir::dyn_cast<mlir::egraph::ReturnOp>(
      egraph.getBody().front().getTerminator());
  if (!returnOp)
    return egraph.emitOpError("expected symbolic fixed-point terminator");

  mlir::egraph::EGraph graph;
  if (mlir::failed(graph.indexEGraph(egraph)))
    return module.emitError("failed to index symbolic fixed-point egraph");

  graph.clearTouchedEClasses();
  mlir::FailureOr<mlir::egraph::EGraphUnionResult> unioned =
      graph.unionValues(graph.getValue(lhs.getSymNameAttr()),
                        graph.getValue(rhs.getSymNameAttr()));
  if (mlir::failed(unioned) || !unioned->changed)
    return rhs.emitOpError("failed to seed symbolic fixed-point union");

  mlir::StringAttr parentRhsName = parentRhs.getSymNameAttr();
  mlir::StringAttr grandRhsName = grandRhs.getSymNameAttr();
  mlir::FailureOr<mlir::egraph::EGraphRebuildResult> rebuilt =
      graph.rebuild(egraph);
  if (mlir::failed(rebuilt))
    return module.emitError("symbolic fixed-point rebuild failed");

  lhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("lhs");
  parentLhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("parent_lhs");
  grandLhs = egraph.lookupSymbol<mlir::egraph::EClassOp>("grand_lhs");
  probe = egraph.lookupSymbol<mlir::egraph::EClassOp>("probe");
  selfCycle = egraph.lookupSymbol<mlir::egraph::EClassOp>("self_cycle");
  mutualA = egraph.lookupSymbol<mlir::egraph::EClassOp>("mutual_a");
  mutualB = egraph.lookupSymbol<mlir::egraph::EClassOp>("mutual_b");
  auto rhsAfter = egraph.lookupSymbol<mlir::egraph::EClassOp>("rhs");
  auto parentRhsAfter =
      egraph.lookupSymbol<mlir::egraph::EClassOp>("parent_rhs");
  auto grandRhsAfter = egraph.lookupSymbol<mlir::egraph::EClassOp>("grand_rhs");
  if (!lhs || !parentLhs || !grandLhs || !probe || !selfCycle || !mutualA ||
      !mutualB)
    return module.emitError("symbolic fixed-point rebuild lost test symbols");
  if (rhsAfter || parentRhsAfter || grandRhsAfter)
    return module.emitError(
        "symbolic fixed-point rebuild did not remove absorbed member eclasses");

  if (graph.getValue(parentRhsName).getLeader().getSymbolNameAttr() !=
      parentLhs.getSymNameAttr())
    return module.emitError(
        "symbolic rebuild did not merge parent eclasses after child union");
  parentLhs.emitRemark()
      << "symbolic rebuild merged parent eclasses after child union";

  if (graph.getValue(grandRhsName).getLeader().getSymbolNameAttr() !=
      grandLhs.getSymNameAttr())
    return module.emitError(
        "symbolic rebuild did not reach grandparent fixed point");
  grandLhs.emitRemark() << "symbolic rebuild reached grandparent fixed point";

  mlir::FailureOr<mlir::StringAttr> parentChildRef =
      getSingleCandidateRef(parentLhs, 1);
  mlir::FailureOr<mlir::StringAttr> grandChildRef =
      getSingleCandidateRef(grandLhs, 1);
  if (mlir::failed(parentChildRef) || *parentChildRef != lhs.getSymNameAttr() ||
      mlir::failed(grandChildRef) ||
      *grandChildRef != parentLhs.getSymNameAttr())
    return parentLhs.emitOpError(
        "symbolic rebuild did not rewrite child refs to leader symbols");
  parentLhs.emitRemark()
      << "symbolic rebuild rewrote child refs to leader symbols";

  if (returnOp.getTargets().size() != 1 ||
      mlir::cast<mlir::FlatSymbolRefAttr>(returnOp.getTargets()[0]).getAttr() !=
          grandLhs.getSymNameAttr())
    return returnOp.emitOpError(
        "symbolic fixed-point rebuild did not retarget return");

  if (parentLhs.getCandidates().size() < 2 ||
      grandLhs.getCandidates().size() < 2)
    return grandLhs.emitOpError(
        "symbolic fixed-point rebuild did not physically absorb transitive "
        "members");
  grandLhs.emitRemark()
      << "symbolic rebuild removed transitive member eclasses";

  mlir::FailureOr<mlir::egraph::EOpRefBase> probeRef =
      getUniqueLiveRootForEClass(graph, probe);
  if (mlir::failed(probeRef))
    return probe.emitOpError("expected a unique live probe candidate root");

  mlir::FailureOr<mlir::egraph::EGraphStructuralKey> probeKey =
      graph.getStructuralKey(*probeRef);
  if (mlir::failed(probeKey))
    return probe.emitOpError("failed to rebuild probe structural key");

  if (probeKey->childLeaderSymbols.size() != 2 ||
      !probeKey->childKeyValues.empty() ||
      probeKey->childLeaderSymbols[0] != lhs.getSymNameAttr() ||
      probeKey->childLeaderSymbols[1] != inputX.getSymNameAttr())
    return probe.emitOpError(
        "symbolic structural key did not hash rebuilt probe by leader symbols");
  probe.emitRemark() << "symbolic structural key hashed rebuilt probe by "
                        "leader symbols";

  mlir::FailureOr<mlir::StringAttr> selfRef =
      getSingleCandidateRef(selfCycle, 0);
  mlir::FailureOr<mlir::StringAttr> mutualARef =
      getSingleCandidateRef(mutualA, 0);
  mlir::FailureOr<mlir::StringAttr> mutualBRef =
      getSingleCandidateRef(mutualB, 0);
  if (mlir::failed(selfRef) || *selfRef != selfCycle.getSymNameAttr() ||
      mlir::failed(mutualARef) || *mutualARef != mutualB.getSymNameAttr() ||
      mlir::failed(mutualBRef) || *mutualBRef != mutualA.getSymNameAttr())
    return selfCycle.emitOpError(
        "symbolic self or mutual cycles were not preserved");
  selfCycle.emitRemark() << "symbolic self and mutual cycles survived rebuild";

  return mlir::success();
}

llvm::StringRef getOpAMode(mlir::egraph::EOpRefBase ref) {
  auto mode = llvm::dyn_cast_or_null<mlir::StringAttr>(
      ref.getOperation()->getAttr("mode"));
  if (!mode)
    return {};
  return mode.getValue();
}

struct TestOpAPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::egraph::test::OpAOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::egraph::test::OpAOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    (void)rewriter;
    root.getOperation()->emitRemark()
        << "typed pattern matched " << root.getOperationName();
    return mlir::success();
  }
};

struct TestSplitPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::egraph::test::SplitOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::egraph::test::SplitOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    (void)rewriter;
    root.getOperation()->emitRemark()
        << "typed pattern matched " << root.getOperationName();
    return mlir::success();
  }
};

unsigned countLeafOps(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](mlir::egraph::test::LeafOp leaf) {
    (void)leaf;
    ++count;
  });
  return count;
}

mlir::LogicalResult verifyScratchOpBeforeDiscard(
    mlir::egraph::EGraph &graph,
    mlir::egraph::EGraphRewriteTransaction &transaction,
    mlir::Operation *scratchOperation, mlir::Operation *diagnosticAnchor) {
  if (!scratchOperation)
    return diagnosticAnchor->emitOpError("expected a scratch operation");
  if (scratchOperation->getBlock() != transaction.getScratchBlock())
    return scratchOperation->emitOpError(
        "expected insertion into the transaction scratch block");
  if (graph.getOperationOwnership(scratchOperation) !=
      mlir::egraph::OperationOwnership::ScratchCreated)
    return scratchOperation->emitOpError(
        "expected scratch-created ownership before discard");

  diagnosticAnchor->emitRemark()
      << "scratch op created in transaction-owned detached block";
  return mlir::success();
}

mlir::LogicalResult
verifyScratchDiscard(mlir::egraph::EGraph &graph,
                     mlir::egraph::EGraphRewriteTransaction &transaction,
                     mlir::Operation *scratchOperation,
                     mlir::Operation *diagnosticAnchor,
                     llvm::StringRef remark) {
  transaction.discard();
  if (!transaction.isDiscarded() || transaction.getScratchBlock())
    return diagnosticAnchor->emitOpError(
        "scratch transaction was not released");
  if (graph.getOperationOwnership(scratchOperation) !=
      mlir::egraph::OperationOwnership::IllegalExternal)
    return diagnosticAnchor->emitOpError(
        "scratch operation remained registered after discard");

  diagnosticAnchor->emitRemark() << remark;
  return mlir::success();
}

mlir::LogicalResult verifyNoPersistentScratch(mlir::ModuleOp module,
                                              mlir::egraph::EGraph &graph,
                                              unsigned initialRefCount,
                                              unsigned initialLeafCount,
                                              llvm::StringRef remark) {
  if (graph.getOpRefs().size() != initialRefCount)
    return module.emitError("scratch rewrite changed egraph occurrences");
  if (countLeafOps(module) != initialLeafCount)
    return module.emitError("scratch rewrite inserted into persistent IR");

  module.emitRemark() << remark;
  return mlir::success();
}

struct ScratchFailurePattern final
    : public mlir::egraph::EGraphPatternFor<mlir::egraph::test::OpAOp> {
  explicit ScratchFailurePattern(
      llvm::SmallVectorImpl<mlir::Operation *> &scratchOperations)
      : scratchOperations(scratchOperations) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::egraph::test::OpAOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::Operation *operation = root.getOperation();
    auto scratch = mlir::egraph::test::LeafOp::create(
        rewriter, root.getLoc(), operation->getResult(0).getType(),
        rewriter.getStringAttr("failure"));
    scratchOperations.push_back(scratch.getOperation());
    return mlir::failure();
  }

private:
  llvm::SmallVectorImpl<mlir::Operation *> &scratchOperations;
};

struct ScratchNoEventPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::egraph::test::OpBOp> {
  explicit ScratchNoEventPattern(
      llvm::SmallVectorImpl<mlir::Operation *> &scratchOperations)
      : scratchOperations(scratchOperations) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::egraph::test::OpBOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::Operation *operation = root.getOperation();
    auto scratch = mlir::egraph::test::LeafOp::create(
        rewriter, root.getLoc(), operation->getResult(0).getType(),
        rewriter.getStringAttr("success-no-event"));
    scratchOperations.push_back(scratch.getOperation());
    return mlir::success();
  }

private:
  llvm::SmallVectorImpl<mlir::Operation *> &scratchOperations;
};

bool hasYieldUse(mlir::Operation *operation) {
  for (mlir::Value result : operation->getResults()) {
    for (mlir::Operation *user : result.getUsers())
      if (llvm::isa<mlir::egraph::YieldOp>(user))
        return true;
  }
  return false;
}

mlir::egraph::test::LeafOp findExternalLeaf(mlir::ModuleOp module) {
  mlir::egraph::test::LeafOp result;
  module.walk([&](mlir::egraph::test::LeafOp leaf) {
    if (!leaf->getParentOfType<mlir::egraph::EClassOp>() && !result)
      result = leaf;
  });
  return result;
}

mlir::LogicalResult verifyRewriteEventRecording(mlir::ModuleOp module,
                                                mlir::egraph::EGraph &graph) {
  mlir::egraph::EOpRefBase opARef;
  mlir::egraph::EOpRefBase opBRef;
  for (mlir::egraph::EOpRefBase ref : graph.getOpRefs()) {
    if (!opARef && llvm::isa<mlir::egraph::test::OpAOp>(ref.getOperation()))
      opARef = ref;
    if (!opBRef && llvm::isa<mlir::egraph::test::OpBOp>(ref.getOperation()))
      opBRef = ref;
  }

  if (!opARef)
    return module.emitError("expected an op_a candidate");
  if (!opBRef)
    return module.emitError("expected an op_b candidate");

  unsigned initialRefCount = graph.getOpRefs().size();
  unsigned initialLeafCount = countLeafOps(module);
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  auto scratchFromRef = mlir::egraph::test::LeafOp::create(
      rewriter, opARef.getLoc(), opARef.getOperation()->getResult(0).getType(),
      rewriter.getStringAttr("replacement-ref"));
  if (mlir::failed(rewriter.replaceOp(
          opARef, scratchFromRef.getOperation()->getResults())))
    return opARef.getOperation()->emitOpError(
        "failed to record EOpRef replacement");

  llvm::ArrayRef<mlir::egraph::EGraphRewriteEvent> rewriteEvents =
      transaction.getRewriteEvents();
  if (rewriteEvents.size() != 1 ||
      rewriteEvents.front().kind !=
          mlir::egraph::EGraphRewriteEventKind::ReplaceOp ||
      rewriteEvents.front().root != opARef ||
      rewriteEvents.front().replacementValues.size() != 1 ||
      rewriteEvents.front().replacementValues.front() !=
          scratchFromRef.getResult())
    return opARef.getOperation()->emitOpError(
        "unexpected EOpRef replacement event");
  opARef.getOperation()->emitRemark()
      << "replacement event recorded from EOpRef";

  auto scratchFromOperation = mlir::egraph::test::LeafOp::create(
      rewriter, opARef.getLoc(), opARef.getOperation()->getResult(0).getType(),
      rewriter.getStringAttr("replacement-operation"));
  if (mlir::failed(rewriter.replaceOp(
          opARef.getOperation(),
          scratchFromOperation.getOperation()->getResults())))
    return opARef.getOperation()->emitOpError(
        "failed to record Operation replacement");

  rewriteEvents = transaction.getRewriteEvents();
  if (rewriteEvents.size() != 2 ||
      rewriteEvents.back().kind !=
          mlir::egraph::EGraphRewriteEventKind::ReplaceOp ||
      rewriteEvents.back().root != opARef ||
      rewriteEvents.back().replacementValues.size() != 1 ||
      rewriteEvents.back().replacementValues.front() !=
          scratchFromOperation.getResult())
    return opARef.getOperation()->emitOpError(
        "unexpected Operation replacement event");
  opARef.getOperation()->emitRemark()
      << "replacement event recorded from Operation lookup";

  if (mlir::succeeded(rewriter.replaceOp(
          scratchFromOperation.getOperation(),
          scratchFromOperation.getOperation()->getResults())))
    return scratchFromOperation.emitOpError(
        "scratch oldOp replacement was accepted");
  if (transaction.getRewriteEvents().size() != 2)
    return scratchFromOperation.emitOpError(
        "rejected scratch oldOp changed replacement events");
  scratchFromOperation.emitRemark() << "scratch oldOp replacement rejected";

  if (mlir::failed(rewriter.replaceAllUsesWith(opBRef, opARef)))
    return opBRef.getOperation()->emitOpError(
        "failed to record EOpRef equivalence");

  rewriteEvents = transaction.getRewriteEvents();
  if (rewriteEvents.size() != 3 ||
      rewriteEvents.back().kind !=
          mlir::egraph::EGraphRewriteEventKind::ReplaceAllUsesWith ||
      rewriteEvents.back().root != opBRef ||
      rewriteEvents.back().replacementValues.size() != 1 ||
      rewriteEvents.back().replacementValues.front() !=
          opARef.getOperation()->getResult(0))
    return opBRef.getOperation()->emitOpError(
        "unexpected EOpRef equivalence event");
  opBRef.getOperation()->emitRemark()
      << "equivalence event recorded from EOpRef";

  if (mlir::failed(rewriter.replaceAllUsesWith(opBRef.getOperation(),
                                               scratchFromRef.getOperation())))
    return opBRef.getOperation()->emitOpError(
        "failed to record scratch target equivalence");

  rewriteEvents = transaction.getRewriteEvents();
  if (rewriteEvents.size() != 4 ||
      rewriteEvents.back().kind !=
          mlir::egraph::EGraphRewriteEventKind::ReplaceAllUsesWith ||
      rewriteEvents.back().root != opBRef ||
      rewriteEvents.back().replacementValues.size() != 1 ||
      rewriteEvents.back().replacementValues.front() !=
          scratchFromRef.getResult())
    return opBRef.getOperation()->emitOpError(
        "unexpected scratch target equivalence event");
  opBRef.getOperation()->emitRemark()
      << "equivalence event recorded with scratch target";

  if (!transaction.hasEvents())
    return module.emitError("expected recorded transaction events");
  if (graph.getOpRefs().size() != initialRefCount)
    return module.emitError("event recording changed egraph occurrences");
  if (countLeafOps(module) != initialLeafCount)
    return module.emitError("event recording inserted into persistent IR");
  if (!hasYieldUse(opARef.getOperation()) ||
      !hasYieldUse(opBRef.getOperation()))
    return module.emitError("event recording performed SSA replacement");

  module.emitRemark() << "event recording left persistent egraph unchanged";
  transaction.discard();
  return mlir::success();
}

mlir::LogicalResult
findRewriteValidationRefs(mlir::ModuleOp module, mlir::egraph::EGraph &graph,
                          mlir::egraph::EOpRefBase &opARef,
                          mlir::egraph::EOpRefBase &opBRef) {
  for (mlir::egraph::EOpRefBase ref : graph.getOpRefs()) {
    if (!opARef && llvm::isa<mlir::egraph::test::OpAOp>(ref.getOperation()))
      opARef = ref;
    if (!opBRef && llvm::isa<mlir::egraph::test::OpBOp>(ref.getOperation()))
      opBRef = ref;
  }

  if (!opARef)
    return module.emitError("expected an op_a candidate");
  if (!opBRef)
    return module.emitError("expected an op_b candidate");
  return mlir::success();
}

mlir::LogicalResult validateScratchReplacement(mlir::ModuleOp module,
                                               mlir::egraph::EGraph &graph,
                                               mlir::egraph::EOpRefBase opRef) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  auto scratch = mlir::egraph::test::LeafOp::create(
      rewriter, opRef.getLoc(), opRef.getOperation()->getResult(0).getType(),
      rewriter.getStringAttr("valid-scratch-replacement"));
  if (mlir::failed(rewriter.replaceOp(opRef, scratch->getResults())))
    return opRef.getOperation()->emitOpError(
        "failed to record valid scratch replacement");
  if (mlir::failed(transaction.validateEvents()))
    return opRef.getOperation()->emitOpError(
        "valid scratch replacement failed validation");

  opRef.getOperation()->emitRemark()
      << "event validation accepted scratch replacement value";
  return mlir::success();
}

mlir::LogicalResult validateEGraphOwnedReplacement(
    mlir::ModuleOp module, mlir::egraph::EGraph &graph,
    mlir::egraph::EOpRefBase oldRef, mlir::egraph::EOpRefBase replacementRef) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  if (mlir::failed(rewriter.replaceOp(
          oldRef, replacementRef.getOperation()->getResults())))
    return oldRef.getOperation()->emitOpError(
        "failed to record valid egraph-owned replacement");
  if (mlir::failed(transaction.validateEvents()))
    return oldRef.getOperation()->emitOpError(
        "valid egraph-owned replacement failed validation");

  oldRef.getOperation()->emitRemark()
      << "event validation accepted egraph-owned replacement value";
  return mlir::success();
}

mlir::LogicalResult validateAliasReplacement(mlir::ModuleOp module,
                                             mlir::egraph::EGraph &graph,
                                             mlir::egraph::EOpRefBase opRef) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  mlir::Value aliasValue = opRef.getOperation()->getOperand(0);
  llvm::SmallVector<mlir::Value> replacements = {aliasValue};
  if (mlir::failed(rewriter.replaceOp(opRef, replacements)))
    return opRef.getOperation()->emitOpError(
        "failed to record valid alias replacement");
  if (mlir::failed(transaction.validateEvents()))
    return opRef.getOperation()->emitOpError(
        "valid alias replacement failed validation");

  opRef.getOperation()->emitRemark()
      << "event validation accepted legal alias replacement value";
  return mlir::success();
}

mlir::LogicalResult validateEGraphOwnedEquivalence(
    mlir::ModuleOp module, mlir::egraph::EGraph &graph,
    mlir::egraph::EOpRefBase fromRef, mlir::egraph::EOpRefBase toRef) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  if (mlir::failed(rewriter.replaceAllUsesWith(fromRef, toRef)))
    return fromRef.getOperation()->emitOpError(
        "failed to record valid egraph-owned equivalence");
  if (mlir::failed(transaction.validateEvents()))
    return fromRef.getOperation()->emitOpError(
        "valid egraph-owned equivalence failed validation");

  fromRef.getOperation()->emitRemark()
      << "event validation accepted egraph-owned equivalence target";
  return mlir::success();
}

mlir::LogicalResult
validateScratchEquivalence(mlir::ModuleOp module, mlir::egraph::EGraph &graph,
                           mlir::egraph::EOpRefBase fromRef) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  auto scratch = mlir::egraph::test::LeafOp::create(
      rewriter, fromRef.getLoc(),
      fromRef.getOperation()->getResult(0).getType(),
      rewriter.getStringAttr("valid-scratch-equivalence"));
  if (mlir::failed(rewriter.replaceAllUsesWith(fromRef.getOperation(),
                                               scratch.getOperation())))
    return fromRef.getOperation()->emitOpError(
        "failed to record valid scratch equivalence");
  if (mlir::failed(transaction.validateEvents()))
    return fromRef.getOperation()->emitOpError(
        "valid scratch equivalence failed validation");

  fromRef.getOperation()->emitRemark()
      << "event validation accepted scratch equivalence target";
  return mlir::success();
}

mlir::LogicalResult
rejectReplacementCountMismatch(mlir::ModuleOp module,
                               mlir::egraph::EGraph &graph,
                               mlir::egraph::EOpRefBase opRef) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  mlir::Type resultType = opRef.getOperation()->getResult(0).getType();
  auto first = mlir::egraph::test::LeafOp::create(
      rewriter, opRef.getLoc(), resultType,
      rewriter.getStringAttr("count-mismatch-first"));
  auto second = mlir::egraph::test::LeafOp::create(
      rewriter, opRef.getLoc(), resultType,
      rewriter.getStringAttr("count-mismatch-second"));
  llvm::SmallVector<mlir::Value> replacements = {first.getResult(),
                                                 second.getResult()};
  if (mlir::failed(rewriter.replaceOp(opRef, replacements)))
    return opRef.getOperation()->emitOpError(
        "failed to record count mismatch replacement");
  if (mlir::succeeded(transaction.validateEvents()))
    return opRef.getOperation()->emitOpError(
        "count mismatch replacement passed validation");

  opRef.getOperation()->emitRemark()
      << "event validation rejected replacement result count mismatch";
  return mlir::success();
}

mlir::LogicalResult
rejectReplacementTypeMismatch(mlir::ModuleOp module,
                              mlir::egraph::EGraph &graph,
                              mlir::egraph::EOpRefBase opRef) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  auto scratch = mlir::egraph::test::LeafOp::create(
      rewriter, opRef.getLoc(), rewriter.getI64Type(),
      rewriter.getStringAttr("type-mismatch"));
  if (mlir::failed(rewriter.replaceOp(opRef, scratch->getResults())))
    return opRef.getOperation()->emitOpError(
        "failed to record type mismatch replacement");
  if (mlir::succeeded(transaction.validateEvents()))
    return opRef.getOperation()->emitOpError(
        "type mismatch replacement passed validation");

  opRef.getOperation()->emitRemark()
      << "event validation rejected replacement type mismatch";
  return mlir::success();
}

mlir::LogicalResult rejectExternalReplacementValue(
    mlir::ModuleOp module, mlir::egraph::EGraph &graph,
    mlir::egraph::EOpRefBase opRef, mlir::egraph::test::LeafOp external) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  if (mlir::failed(rewriter.replaceOp(opRef, external->getResults())))
    return opRef.getOperation()->emitOpError(
        "failed to record external replacement value");
  if (mlir::succeeded(transaction.validateEvents()))
    return opRef.getOperation()->emitOpError(
        "external replacement value passed validation");

  opRef.getOperation()->emitRemark()
      << "event validation rejected illegal external replacement value";
  return mlir::success();
}

mlir::LogicalResult rejectExternalEquivalenceTarget(
    mlir::ModuleOp module, mlir::egraph::EGraph &graph,
    mlir::egraph::EOpRefBase fromRef, mlir::egraph::test::LeafOp external) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  if (mlir::succeeded(rewriter.replaceAllUsesWith(fromRef.getOperation(),
                                                  external.getOperation())))
    return fromRef.getOperation()->emitOpError(
        "external equivalence target was accepted");
  if (!transaction.getRewriteEvents().empty())
    return fromRef.getOperation()->emitOpError(
        "external equivalence target recorded an event");

  fromRef.getOperation()->emitRemark()
      << "illegal external equivalence target rejected before validation";
  return mlir::success();
}

mlir::LogicalResult
rejectScratchOldReplacement(mlir::ModuleOp module, mlir::egraph::EGraph &graph,
                            mlir::egraph::EOpRefBase opRef) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  auto scratch = mlir::egraph::test::LeafOp::create(
      rewriter, opRef.getLoc(), opRef.getOperation()->getResult(0).getType(),
      rewriter.getStringAttr("scratch-old"));
  if (mlir::succeeded(
          rewriter.replaceOp(scratch.getOperation(), scratch->getResults())))
    return opRef.getOperation()->emitOpError(
        "scratch oldOp replacement was accepted");
  if (!transaction.getRewriteEvents().empty())
    return opRef.getOperation()->emitOpError(
        "scratch oldOp replacement recorded an event");

  opRef.getOperation()->emitRemark()
      << "scratch oldOp replacement rejected before validation";
  return mlir::success();
}

mlir::LogicalResult verifyRewriteEventValidation(mlir::ModuleOp module,
                                                 mlir::egraph::EGraph &graph) {
  mlir::egraph::EOpRefBase opARef;
  mlir::egraph::EOpRefBase opBRef;
  if (mlir::failed(findRewriteValidationRefs(module, graph, opARef, opBRef)))
    return mlir::failure();

  mlir::egraph::test::LeafOp external = findExternalLeaf(module);
  if (!external)
    return module.emitError("expected an external egraph_test.leaf op");

  if (mlir::failed(validateScratchReplacement(module, graph, opARef)) ||
      mlir::failed(
          validateEGraphOwnedReplacement(module, graph, opARef, opBRef)) ||
      mlir::failed(validateAliasReplacement(module, graph, opARef)) ||
      mlir::failed(
          validateEGraphOwnedEquivalence(module, graph, opBRef, opARef)) ||
      mlir::failed(validateScratchEquivalence(module, graph, opBRef)) ||
      mlir::failed(rejectReplacementCountMismatch(module, graph, opARef)) ||
      mlir::failed(rejectReplacementTypeMismatch(module, graph, opARef)) ||
      mlir::failed(
          rejectExternalReplacementValue(module, graph, opARef, external)) ||
      mlir::failed(
          rejectExternalEquivalenceTarget(module, graph, opBRef, external)) ||
      mlir::failed(rejectScratchOldReplacement(module, graph, opARef)))
    return mlir::failure();

  return mlir::success();
}

enum class NegativeFailureScenario {
  TypeMismatch,
  IllegalExternalValue,
  IllegalExternalOp,
  ScratchOldOp,
};

mlir::FailureOr<NegativeFailureScenario>
getNegativeFailureScenario(mlir::ModuleOp module) {
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    llvm::StringRef name = func.getName();
    if (name == "reject_transaction_type_mismatch")
      return NegativeFailureScenario::TypeMismatch;
    if (name == "reject_illegal_external_value")
      return NegativeFailureScenario::IllegalExternalValue;
    if (name == "reject_illegal_external_op")
      return NegativeFailureScenario::IllegalExternalOp;
    if (name == "reject_scratch_old_op")
      return NegativeFailureScenario::ScratchOldOp;
  }

  module.emitError("expected a recognized egraph negative test function");
  return mlir::failure();
}

mlir::LogicalResult emitRejectedTypeMismatch(mlir::ModuleOp module,
                                             mlir::egraph::EGraph &graph,
                                             mlir::egraph::EOpRefBase opRef) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  auto scratch = mlir::egraph::test::LeafOp::create(
      rewriter, opRef.getLoc(), rewriter.getI64Type(),
      rewriter.getStringAttr("negative-type-mismatch"));
  if (mlir::failed(rewriter.replaceOp(opRef, scratch->getResults())))
    return opRef.getOperation()->emitOpError(
        "negative test failed to record replacement type mismatch");
  if (mlir::succeeded(transaction.validateEvents()))
    return opRef.getOperation()->emitOpError(
        "negative test accepted replacement type mismatch");

  return opRef.getOperation()->emitOpError(
      "negative test rejected replacement type mismatch");
}

mlir::LogicalResult emitRejectedExternalReplacementValue(
    mlir::ModuleOp module, mlir::egraph::EGraph &graph,
    mlir::egraph::EOpRefBase opRef, mlir::egraph::test::LeafOp external) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  if (mlir::failed(rewriter.replaceOp(opRef, external->getResults())))
    return opRef.getOperation()->emitOpError(
        "negative test failed to record illegal external replacement value");
  if (mlir::succeeded(transaction.validateEvents()))
    return opRef.getOperation()->emitOpError(
        "negative test accepted illegal external replacement value");

  return opRef.getOperation()->emitOpError(
      "negative test rejected illegal external replacement value");
}

mlir::LogicalResult emitRejectedExternalEquivalenceTarget(
    mlir::ModuleOp module, mlir::egraph::EGraph &graph,
    mlir::egraph::EOpRefBase fromRef, mlir::egraph::test::LeafOp external) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  if (mlir::failed(rewriter.replaceAllUsesWith(fromRef.getOperation(),
                                               external.getOperation())) &&
      transaction.getRewriteEvents().empty())
    return fromRef.getOperation()->emitOpError(
        "negative test rejected illegal external equivalence target");

  return fromRef.getOperation()->emitOpError(
      "negative test accepted illegal external equivalence target");
}

mlir::LogicalResult emitRejectedScratchOldOp(mlir::ModuleOp module,
                                             mlir::egraph::EGraph &graph,
                                             mlir::egraph::EOpRefBase opRef) {
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  auto scratch = mlir::egraph::test::LeafOp::create(
      rewriter, opRef.getLoc(), opRef.getOperation()->getResult(0).getType(),
      rewriter.getStringAttr("negative-scratch-old"));
  if (mlir::failed(
          rewriter.replaceOp(scratch.getOperation(), scratch->getResults())) &&
      transaction.getRewriteEvents().empty())
    return opRef.getOperation()->emitOpError(
        "negative test rejected scratch oldOp replacement");

  return opRef.getOperation()->emitOpError(
      "negative test accepted scratch oldOp replacement");
}

mlir::LogicalResult verifyNegativeFailures(mlir::ModuleOp module,
                                           mlir::egraph::EGraph &graph) {
  mlir::FailureOr<NegativeFailureScenario> scenario =
      getNegativeFailureScenario(module);
  if (mlir::failed(scenario))
    return mlir::failure();

  mlir::egraph::EOpRefBase opARef;
  mlir::egraph::EOpRefBase opBRef;
  if (mlir::failed(findRewriteValidationRefs(module, graph, opARef, opBRef)))
    return mlir::failure();

  switch (*scenario) {
  case NegativeFailureScenario::TypeMismatch:
    return emitRejectedTypeMismatch(module, graph, opARef);
  case NegativeFailureScenario::IllegalExternalValue: {
    mlir::egraph::test::LeafOp external = findExternalLeaf(module);
    if (!external)
      return module.emitError("expected an external egraph_test.leaf op");
    return emitRejectedExternalReplacementValue(module, graph, opARef,
                                                external);
  }
  case NegativeFailureScenario::IllegalExternalOp: {
    mlir::egraph::test::LeafOp external = findExternalLeaf(module);
    if (!external)
      return module.emitError("expected an external egraph_test.leaf op");
    return emitRejectedExternalEquivalenceTarget(module, graph, opBRef,
                                                 external);
  }
  case NegativeFailureScenario::ScratchOldOp:
    return emitRejectedScratchOldOp(module, graph, opARef);
  }

  llvm_unreachable("unknown egraph negative failure scenario");
}

unsigned countEClasses(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](mlir::egraph::EClassOp) { ++count; });
  return count;
}

mlir::FailureOr<mlir::egraph::EOpRef<mlir::arith::MulIOp>>
findMulRef(mlir::egraph::EGraph &graph) {
  for (mlir::egraph::EOpRefBase ref : graph.getOpRefs())
    if (llvm::isa<mlir::arith::MulIOp>(ref.getOperation()))
      return mlir::egraph::EOpRef<mlir::arith::MulIOp>(ref);
  return mlir::failure();
}

bool containsShli(llvm::ArrayRef<mlir::egraph::EOpRefBase> refs) {
  return llvm::any_of(refs, [](mlir::egraph::EOpRefBase ref) {
    return llvm::isa<mlir::arith::ShLIOp>(ref.getOperation());
  });
}

bool containsAddi(llvm::ArrayRef<mlir::egraph::EOpRefBase> refs) {
  return llvm::any_of(refs, [](mlir::egraph::EOpRefBase ref) {
    return llvm::isa<mlir::arith::AddIOp>(ref.getOperation());
  });
}

bool containsConstantOne(llvm::ArrayRef<mlir::egraph::EOpRefBase> refs) {
  return llvm::any_of(refs, [](mlir::egraph::EOpRefBase ref) {
    auto constant = llvm::dyn_cast<mlir::arith::ConstantOp>(ref.getOperation());
    auto integer = constant
                       ? llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                       : mlir::IntegerAttr();
    return integer && integer.getInt() == 1;
  });
}

mlir::LogicalResult
verifySymbolicTransactionCommit(mlir::ModuleOp module,
                                mlir::egraph::EGraph &graph) {
  mlir::FailureOr<mlir::egraph::EOpRef<mlir::arith::MulIOp>> mulRef =
      findMulRef(graph);
  if (mlir::failed(mulRef))
    return module.emitError("expected one symbolic arith.muli candidate");

  unsigned initialEClassCount = countEClasses(module);
  mlir::egraph::EValue oldValue = mulRef->getResult(0);
  mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                     module.getContext());
  mlir::egraph::EGraphPatternRewriter rewriter(transaction);

  auto resultType = llvm::dyn_cast<mlir::IntegerType>(
      mulRef->getOperation()->getResult(0).getType());
  if (!resultType)
    return mulRef->getOperation()->emitOpError("expected integer mul result");

  auto one = mlir::arith::ConstantOp::create(
      rewriter, mulRef->getLoc(), rewriter.getIntegerAttr(resultType, 1));
  auto shifted = mlir::arith::ShLIOp::create(
      rewriter, mulRef->getLoc(), mulRef->getOperation()->getOperand(0),
      one.getResult());
  mlir::Operation *oneOperation = one.getOperation();
  mlir::Operation *shiftedOperation = shifted.getOperation();

  if (mlir::failed(rewriter.replaceOp(*mulRef, shifted->getResults())))
    return mulRef->getOperation()->emitOpError(
        "failed to record symbolic scratch DAG replacement");

  mlir::FailureOr<mlir::egraph::EGraphRewriteCommitResult> committed =
      transaction.commit(module);
  if (mlir::failed(committed) || !committed->changed)
    return module.emitError("symbolic transaction commit did not change");
  if (!transaction.isDiscarded() || transaction.getScratchBlock())
    return module.emitError("symbolic transaction commit leaked scratch block");
  if (graph.getOperationOwnership(oneOperation) !=
          mlir::egraph::OperationOwnership::IllegalExternal ||
      graph.getOperationOwnership(shiftedOperation) !=
          mlir::egraph::OperationOwnership::IllegalExternal)
    return module.emitError(
        "symbolic transaction commit left scratch ownership");
  if (mulRef->isLive())
    return module.emitError(
        "symbolic transaction commit left old mul ref live");
  if (countEClasses(module) != initialEClassCount + 1)
    return module.emitError(
        "symbolic transaction commit did not preserve the constant scratch "
        "eclass");
  if (!containsShli(committed->newCandidateRoots) ||
      !containsConstantOne(committed->newCandidateRoots))
    return module.emitError(
        "symbolic transaction commit did not expose new shli/constant roots");
  if (!containsAddi(committed->affectedParentCandidates))
    return module.emitError(
        "symbolic transaction commit lost affected parent candidate");

  bool sawShiftedDef = false;
  for (mlir::egraph::EOpRef<mlir::arith::ShLIOp> ref :
       oldValue.getDefs<mlir::arith::ShLIOp>()) {
    mlir::FailureOr<mlir::egraph::EOpRef<mlir::arith::ConstantOp>> constant =
        ref.getOperand(1).getUniqueDef<mlir::arith::ConstantOp>();
    if (mlir::failed(constant))
      continue;

    auto integer =
        llvm::dyn_cast<mlir::IntegerAttr>(constant->getOp().getValue());
    if (integer && integer.getInt() == 1)
      sawShiftedDef = true;
  }
  if (!sawShiftedDef)
    return module.emitError(
        "symbolic transaction commit did not merge shli candidate into mul");

  module.emitRemark() << "symbolic commit interned scratch DAG";
  module.emitRemark() << "symbolic commit merged shli candidate into @mul";
  module.emitRemark() << "symbolic commit returned affected parent candidates";
  return mlir::success();
}

mlir::FailureOr<mlir::egraph::EOpRefBase>
findOpBByTag(mlir::egraph::EGraph &graph, llvm::StringRef tag) {
  for (mlir::egraph::EOpRefBase ref : graph.getOpRefs()) {
    auto opB = llvm::dyn_cast<mlir::egraph::test::OpBOp>(ref.getOperation());
    if (!opB)
      continue;

    auto tagAttr = opB->getAttrOfType<mlir::StringAttr>("tag");
    if (tagAttr && tagAttr.getValue() == tag)
      return ref;
  }

  return mlir::failure();
}

struct WorklistFailingPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::egraph::test::OpBOp> {
  explicit WorklistFailingPattern(bool &sawFailedPattern)
      : sawFailedPattern(sawFailedPattern) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::egraph::test::OpBOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    (void)rewriter;
    auto tagAttr = root.getOp()->getAttrOfType<mlir::StringAttr>("tag");
    if (!tagAttr || tagAttr.getValue() != "driver_parent")
      return mlir::failure();

    sawFailedPattern = true;
    return mlir::failure();
  }

private:
  bool &sawFailedPattern;
};

struct WorklistNoOpPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::egraph::test::OpBOp> {
  explicit WorklistNoOpPattern(bool &sawNoOp) : sawNoOp(sawNoOp) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::egraph::test::OpBOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    (void)rewriter;
    auto tagAttr = root.getOp()->getAttrOfType<mlir::StringAttr>("tag");
    if (!tagAttr || tagAttr.getValue() != "driver_parent")
      return mlir::failure();

    sawNoOp = true;
    return mlir::success();
  }

private:
  bool &sawNoOp;
};

struct WorklistNestedRewritePattern final
    : public mlir::egraph::EGraphPatternFor<mlir::egraph::test::OpBOp> {
  explicit WorklistNestedRewritePattern(bool &matchedNested)
      : matchedNested(matchedNested) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::egraph::test::OpBOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    auto tagAttr = root.getOp()->getAttrOfType<mlir::StringAttr>("tag");
    if (!tagAttr || tagAttr.getValue() != "driver_parent")
      return mlir::failure();

    return root.getOperand(0).matchDef<mlir::egraph::test::OpAOp>(
        [&](mlir::egraph::EOpRef<mlir::egraph::test::OpAOp> opA) {
          if (getOpAMode(opA) != "driver_child")
            return mlir::failure();

          mlir::Value lhs = opA.getOperation()->getOperand(0);
          auto replacement = mlir::egraph::test::OpBOp::create(
              rewriter, root.getLoc(),
              root.getOperation()->getResult(0).getType(), lhs,
              rewriter.getStringAttr("driver_added"));
          matchedNested = true;
          root.getOperation()->emitRemark()
              << "worklist driver matched nested op_a";
          return rewriter.replaceOp(root, replacement->getResults());
        });
  }

private:
  bool &matchedNested;
};

struct WorklistAddedCandidatePattern final
    : public mlir::egraph::EGraphPatternFor<mlir::egraph::test::OpBOp> {
  explicit WorklistAddedCandidatePattern(bool &sawAddedCandidate)
      : sawAddedCandidate(sawAddedCandidate) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::egraph::test::OpBOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    (void)rewriter;
    auto tagAttr = root.getOp()->getAttrOfType<mlir::StringAttr>("tag");
    if (!tagAttr || tagAttr.getValue() != "driver_added")
      return mlir::failure();

    sawAddedCandidate = true;
    root.getOperation()->emitRemark()
        << "worklist driver processed enqueued new candidate root";
    return mlir::success();
  }

private:
  bool &sawAddedCandidate;
};

struct WorklistRhsRewritePattern final
    : public mlir::egraph::EGraphPatternFor<mlir::egraph::test::OpBOp> {
  explicit WorklistRhsRewritePattern(bool &rewroteRhs)
      : rewroteRhs(rewroteRhs) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::egraph::test::OpBOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    auto tagAttr = root.getOp()->getAttrOfType<mlir::StringAttr>("tag");
    if (!tagAttr || tagAttr.getValue() != "driver_rhs")
      return mlir::failure();

    auto replacement = mlir::egraph::test::OpBOp::create(
        rewriter, root.getLoc(), root.getOperation()->getResult(0).getType(),
        root.getOperation()->getOperand(0),
        rewriter.getStringAttr("driver_added"));
    rewroteRhs = true;
    return rewriter.replaceOp(root, replacement->getResults());
  }

private:
  bool &rewroteRhs;
};

mlir::LogicalResult verifyWorklistDriverIterationLimit(
    mlir::ModuleOp module, const mlir::egraph::EGraphPatternSet &patterns) {
  mlir::OwningOpRef<mlir::Operation *> clonedModuleRef(module->clone());
  auto clonedModule = mlir::cast<mlir::ModuleOp>(clonedModuleRef.get());

  mlir::egraph::EGraph graph;
  if (mlir::failed(indexTestGraph(clonedModule, graph)))
    return module.emitError("failed to index cloned worklist graph");

  unsigned initialRefCount = graph.getOpRefs().size();
  unsigned initialEClassCount = countEClasses(clonedModule);

  mlir::egraph::EGraphMatchConfig config;
  config.maxIterations = 1;

  mlir::FailureOr<mlir::egraph::EGraphMatchStats> driven =
      mlir::egraph::applyEGraphPatterns(graph, patterns, clonedModule, config);
  if (mlir::failed(driven))
    return module.emitError("limited worklist driver failed");
  if (!driven->limitReached ||
      driven->reachedLimit != mlir::egraph::EGraphMatchLimit::Iteration)
    return module.emitError("worklist driver did not report iteration limit");
  if (driven->iterations != 1)
    return module.emitError("worklist driver used unexpected iteration count");
  if (driven->changed || driven->changedCommits != 0)
    return module.emitError("limited worklist driver committed a rewrite");
  if (driven->rebuilds != 0)
    return module.emitError("limited worklist driver rebuilt unexpectedly");
  if (driven->enqueuedCandidates != initialRefCount ||
      driven->skippedCandidateCap != 0 ||
      graph.getOpRefs().size() != initialRefCount ||
      countEClasses(clonedModule) != initialEClassCount)
    return module.emitError("limited worklist driver changed persistent state");

  module.emitRemark() << "worklist driver reached iteration limit";
  return mlir::success();
}

mlir::LogicalResult verifyWorklistDriverCandidateCap(mlir::ModuleOp module) {
  mlir::OwningOpRef<mlir::Operation *> clonedModuleRef(module->clone());
  auto clonedModule = mlir::cast<mlir::ModuleOp>(clonedModuleRef.get());

  mlir::egraph::EGraph graph;
  if (mlir::failed(indexTestGraph(clonedModule, graph)))
    return module.emitError("failed to index cloned candidate-cap graph");

  bool rewroteRhs = false;
  bool sawAddedCandidate = false;
  mlir::egraph::EGraphPatternSet patterns;
  patterns.add<WorklistRhsRewritePattern>(rewroteRhs);
  patterns.add<WorklistAddedCandidatePattern>(sawAddedCandidate);

  mlir::egraph::EGraphMatchConfig config;
  config.maxCandidatesPerEClass = 1;

  mlir::FailureOr<mlir::egraph::EGraphMatchStats> driven =
      mlir::egraph::applyEGraphPatterns(graph, patterns, clonedModule, config);
  if (mlir::failed(driven))
    return module.emitError("candidate-capped worklist driver failed");
  if (driven->limitReached ||
      driven->reachedLimit != mlir::egraph::EGraphMatchLimit::None)
    return module.emitError("candidate cap unexpectedly stopped the driver");
  if (!driven->changed || driven->changedCommits != 1 || driven->rebuilds != 1)
    return module.emitError("candidate cap changed rewrite commit behavior");
  if (driven->skippedCandidateCap == 0)
    return module.emitError(
        "worklist driver did not skip any per-eclass candidates");
  if (!rewroteRhs)
    return module.emitError("candidate-capped worklist driver missed rewrite");
  if (sawAddedCandidate)
    return module.emitError(
        "candidate-capped worklist driver processed a capped candidate");
  if (mlir::failed(findOpBByTag(graph, "driver_added")))
    return module.emitError(
        "candidate-capped worklist driver did not create the capped candidate");

  module.emitRemark() << "worklist driver skipped per-eclass candidate cap";
  return mlir::success();
}

mlir::LogicalResult verifyWorklistDriverMlirFold(mlir::ModuleOp module) {
  mlir::OwningOpRef<mlir::Operation *> clonedModuleRef(module->clone());
  auto clonedModule = mlir::cast<mlir::ModuleOp>(clonedModuleRef.get());

  mlir::egraph::EGraph graph;
  if (mlir::failed(indexTestGraph(clonedModule, graph)))
    return module.emitError("failed to index cloned MLIR-fold graph");

  mlir::egraph::EGraphPatternSet patterns;
  mlir::egraph::EGraphMatchConfig config;
  config.enableMlirFold = true;

  mlir::FailureOr<mlir::egraph::EGraphMatchStats> driven =
      mlir::egraph::applyEGraphPatterns(graph, patterns, clonedModule, config);
  if (mlir::failed(driven))
    return module.emitError("MLIR-fold worklist driver failed");
  if (driven->limitReached ||
      driven->reachedLimit != mlir::egraph::EGraphMatchLimit::None)
    return module.emitError("MLIR-fold worklist driver unexpectedly stopped");
  if (!driven->changed || driven->changedCommits == 0)
    return module.emitError("MLIR-fold worklist driver did not add a rewrite");
  if (driven->matchedPatterns != 0)
    return module.emitError("MLIR fold was counted as a user pattern");

  mlir::FailureOr<mlir::egraph::EOpRefBase> foldedRoot =
      findOpBByTag(graph, "driver_rhs");
  if (mlir::failed(foldedRoot))
    return module.emitError("expected tagged op_b candidate after MLIR fold");

  bool sawTaggedDef = false;
  bool sawTaglessDef = false;
  for (mlir::egraph::EOpRef<mlir::egraph::test::OpBOp> def :
       foldedRoot->getResult(0).getDefs<mlir::egraph::test::OpBOp>()) {
    auto tagAttr = def.getOp()->getAttrOfType<mlir::StringAttr>("tag");
    sawTaggedDef |= tagAttr && tagAttr.getValue() == "driver_rhs";
    sawTaglessDef |= !tagAttr;
  }
  if (!sawTaggedDef || !sawTaglessDef)
    return module.emitError(
        "MLIR fold did not keep tagged and tagless egraph alternatives");

  module.emitRemark() << "worklist driver added MLIR fold alternative";
  return mlir::success();
}

mlir::LogicalResult verifyWorklistDriver(mlir::ModuleOp module,
                                         mlir::egraph::EGraph &graph) {
  bool sawFailedPattern = false;
  bool sawNoOp = false;
  bool matchedNested = false;
  bool sawAddedCandidate = false;
  mlir::egraph::EGraphPatternSet patterns;
  patterns.add<WorklistFailingPattern>(sawFailedPattern);
  patterns.add<WorklistNoOpPattern>(sawNoOp);
  patterns.add<WorklistNestedRewritePattern>(matchedNested);
  patterns.add<WorklistAddedCandidatePattern>(sawAddedCandidate);

  if (mlir::failed(verifyWorklistDriverIterationLimit(module, patterns)))
    return mlir::failure();

  mlir::FailureOr<mlir::egraph::EGraphMatchStats> driven =
      mlir::egraph::applyEGraphPatterns(graph, patterns, module);
  if (mlir::failed(driven))
    return module.emitError("worklist driver failed");
  if (driven->limitReached ||
      driven->reachedLimit != mlir::egraph::EGraphMatchLimit::None)
    return module.emitError("worklist driver unexpectedly hit a limit");
  if (!driven->changed || driven->changedCommits == 0)
    return module.emitError("worklist driver did not commit a changed rewrite");
  if (driven->changedCommits != 1)
    return module.emitError("worklist driver used unexpected changed commits");
  if (driven->rebuilds != 1)
    return module.emitError(
        "worklist driver did not rebuild once after batched commits");
  if (driven->matchedPatterns <= driven->changedCommits)
    return module.emitError(
        "worklist driver stats did not distinguish matched patterns");
  if (!sawFailedPattern)
    return module.emitError("worklist driver did not exercise failure path");
  if (!sawNoOp)
    return module.emitError("worklist driver did not exercise no-op success");
  if (!matchedNested)
    return module.emitError("worklist driver did not match nested op_a");
  if (!sawAddedCandidate)
    return module.emitError("worklist driver did not process new roots");

  mlir::FailureOr<mlir::egraph::EOpRefBase> parentRef =
      findOpBByTag(graph, "driver_parent");
  if (mlir::failed(parentRef))
    return module.emitError("expected live driver parent candidate");

  bool sawAddedDef = false;
  for (mlir::egraph::EOpRef<mlir::egraph::test::OpBOp> def :
       parentRef->getResult(0).getDefs<mlir::egraph::test::OpBOp>()) {
    auto tagAttr = def.getOp()->getAttrOfType<mlir::StringAttr>("tag");
    sawAddedDef |= tagAttr && tagAttr.getValue() == "driver_added";
  }
  if (!sawAddedDef)
    return module.emitError(
        "worklist driver did not expose the added equivalent candidate");

  emitDriverStatsRemark(module, *driven);
  module.emitRemark() << "worklist driver added equivalent candidate";
  module.emitRemark() << "worklist driver discarded failed pattern";
  module.emitRemark() << "worklist driver skipped rebuild for no-op success";
  module.emitRemark() << "worklist driver rebuilt after batched commits";
  module.emitRemark() << "worklist driver stats counted matched patterns";
  return mlir::success();
}

mlir::LogicalResult verifySymbolicWorklistDriver(mlir::ModuleOp module,
                                                 mlir::egraph::EGraph &graph) {
  mlir::FailureOr<mlir::egraph::EGraphOp> egraph =
      getSingleActiveEGraph(module);
  if (mlir::failed(egraph))
    return module.emitError("expected one active egraph.egraph");
  if (!graph.isClean())
    return egraph->emitOpError(
        "expected indexed symbolic graph to start clean");
  if (mlir::failed(seedDirtyWorklistGraph(module, graph)))
    return mlir::failure();

  bool sawFailedPattern = false;
  bool sawNoOp = false;
  bool matchedNested = false;
  bool sawAddedCandidate = false;
  mlir::egraph::EGraphPatternSet patterns;
  patterns.add<WorklistFailingPattern>(sawFailedPattern);
  patterns.add<WorklistNoOpPattern>(sawNoOp);
  patterns.add<WorklistNestedRewritePattern>(matchedNested);
  patterns.add<WorklistAddedCandidatePattern>(sawAddedCandidate);

  if (mlir::failed(verifyWorklistDriverIterationLimit(module, patterns)) ||
      mlir::failed(verifyWorklistDriverCandidateCap(module)) ||
      mlir::failed(verifyWorklistDriverMlirFold(module)))
    return mlir::failure();

  mlir::FailureOr<mlir::egraph::EGraphMatchStats> driven =
      mlir::egraph::applyEGraphPatterns(graph, patterns, module);
  if (mlir::failed(driven))
    return module.emitError("symbolic worklist driver failed");
  if (!graph.isClean())
    return module.emitError("symbolic worklist driver left a dirty graph");
  if (driven->limitReached ||
      driven->reachedLimit != mlir::egraph::EGraphMatchLimit::None)
    return module.emitError(
        "symbolic worklist driver unexpectedly hit a limit");
  if (!driven->changed || driven->changedCommits == 0)
    return module.emitError(
        "symbolic worklist driver did not commit a changed rewrite");
  if (driven->changedCommits != 1)
    return module.emitError(
        "symbolic worklist driver used unexpected changed commits");
  if (driven->rebuilds != 2)
    return module.emitError(
        "symbolic worklist driver did not count initial and changed rebuilds");
  if (driven->matchedPatterns <= driven->changedCommits)
    return module.emitError(
        "symbolic worklist driver stats did not distinguish matched patterns");
  if (!sawFailedPattern)
    return module.emitError(
        "symbolic worklist driver did not exercise failure path");
  if (!sawNoOp)
    return module.emitError(
        "symbolic worklist driver did not exercise no-op success");
  if (!matchedNested)
    return module.emitError(
        "symbolic worklist driver did not rebuild before matching");
  if (!sawAddedCandidate)
    return module.emitError(
        "symbolic worklist driver did not process new roots");

  mlir::FailureOr<mlir::egraph::EOpRefBase> parentRef =
      findOpBByTag(graph, "driver_parent");
  if (mlir::failed(parentRef))
    return module.emitError("expected live symbolic driver parent candidate");

  bool sawAddedDef = false;
  for (mlir::egraph::EOpRef<mlir::egraph::test::OpBOp> def :
       parentRef->getResult(0).getDefs<mlir::egraph::test::OpBOp>()) {
    auto tagAttr = def.getOp()->getAttrOfType<mlir::StringAttr>("tag");
    sawAddedDef |= tagAttr && tagAttr.getValue() == "driver_added";
  }
  if (!sawAddedDef)
    return module.emitError(
        "symbolic worklist driver did not expose the added equivalent "
        "candidate");

  emitDriverStatsRemark(module, *driven);
  module.emitRemark() << "worklist driver rebuilt dirty graph before matching";
  module.emitRemark() << "worklist driver added equivalent candidate";
  module.emitRemark() << "worklist driver discarded failed pattern";
  module.emitRemark() << "worklist driver skipped rebuild for no-op success";
  module.emitRemark() << "worklist driver rebuilt after batched commits";
  module.emitRemark() << "worklist driver stats counted matched patterns";
  return mlir::success();
}

bool hasConstantIntegerDef(mlir::egraph::EValue value, int64_t expected) {
  for (mlir::egraph::EOpRef<mlir::arith::ConstantOp> def :
       value.getDefs<mlir::arith::ConstantOp>()) {
    auto integer = llvm::dyn_cast<mlir::IntegerAttr>(def.getOp().getValue());
    if (integer && integer.getInt() == expected)
      return true;
  }
  return false;
}

bool hasShiftLeftByOneDef(mlir::egraph::EValue value,
                          mlir::egraph::EValue shiftedValue) {
  for (mlir::egraph::EOpRef<mlir::arith::ShLIOp> def :
       value.getDefs<mlir::arith::ShLIOp>()) {
    if (!def.getOperand(0).isEquivalentTo(shiftedValue) ||
        !hasConstantIntegerDef(def.getOperand(1), 1))
      continue;
    return true;
  }
  return false;
}

bool hasExactDivDef(mlir::egraph::EValue value, mlir::egraph::EValue lhs,
                    mlir::egraph::EValue rhs) {
  for (mlir::egraph::EOpRef<mlir::arith::DivSIOp> def :
       value.getDefs<mlir::arith::DivSIOp>()) {
    if (def.getOperand(0).isEquivalentTo(lhs) &&
        def.getOperand(1).isEquivalentTo(rhs))
      return true;
  }
  return false;
}

bool hasMulOfDivDef(mlir::egraph::EValue value, mlir::egraph::EValue mulLhs,
                    mlir::egraph::EValue divLhs, mlir::egraph::EValue divRhs) {
  for (mlir::egraph::EOpRef<mlir::arith::MulIOp> def :
       value.getDefs<mlir::arith::MulIOp>()) {
    if (!def.getOperand(0).isEquivalentTo(mulLhs) ||
        !hasExactDivDef(def.getOperand(1), divLhs, divRhs))
      continue;
    return true;
  }
  return false;
}

struct DemoMulByTwoToShiftPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::arith::MulIOp> {
  explicit DemoMulByTwoToShiftPattern(unsigned &matchCount)
      : matchCount(matchCount) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::arith::MulIOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    auto resultType = llvm::dyn_cast<mlir::IntegerType>(
        root.getOperation()->getResult(0).getType());
    if (!resultType)
      return mlir::failure();

    mlir::Value shiftedPayload;
    mlir::egraph::EValue shiftedValue;
    if (hasConstantIntegerDef(root.getOperand(0), 2)) {
      shiftedPayload = root.getOperation()->getOperand(1);
      shiftedValue = root.getOperand(1);
    } else if (hasConstantIntegerDef(root.getOperand(1), 2)) {
      shiftedPayload = root.getOperation()->getOperand(0);
      shiftedValue = root.getOperand(0);
    } else {
      return mlir::failure();
    }

    if (hasShiftLeftByOneDef(root.getResult(0), shiftedValue))
      return mlir::failure();

    auto one = mlir::arith::ConstantOp::create(
        rewriter, root.getLoc(), rewriter.getIntegerAttr(resultType, 1));
    auto replacement = mlir::arith::ShLIOp::create(
        rewriter, root.getLoc(), shiftedPayload, one.getResult());
    ++matchCount;
    return rewriter.replaceOp(root, replacement->getResults());
  }

private:
  unsigned &matchCount;
};

struct DemoReassociateDivPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::arith::DivSIOp> {
  explicit DemoReassociateDivPattern(unsigned &matchCount)
      : matchCount(matchCount) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::arith::DivSIOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    return root.getOperand(0).matchDef<mlir::arith::MulIOp>(
        [&](mlir::egraph::EOpRef<mlir::arith::MulIOp> mul) {
          if (hasMulOfDivDef(root.getResult(0), mul.getOperand(0),
                             mul.getOperand(1), root.getOperand(1)))
            return mlir::failure();

          auto rotatedDiv = mlir::arith::DivSIOp::create(
              rewriter, root.getLoc(), mul.getOperation()->getOperand(1),
              root.getOperation()->getOperand(1));
          auto replacement = mlir::arith::MulIOp::create(
              rewriter, root.getLoc(), mul.getOperation()->getOperand(0),
              rotatedDiv.getResult());
          ++matchCount;
          return rewriter.replaceOp(root, replacement->getResults());
        });
  }

private:
  unsigned &matchCount;
};

struct DemoDivSelfToOnePattern final
    : public mlir::egraph::EGraphPatternFor<mlir::arith::DivSIOp> {
  explicit DemoDivSelfToOnePattern(unsigned &matchCount)
      : matchCount(matchCount) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::arith::DivSIOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    auto resultType = llvm::dyn_cast<mlir::IntegerType>(
        root.getOperation()->getResult(0).getType());
    if (!resultType || !root.getOperand(0).isEquivalentTo(root.getOperand(1)) ||
        hasConstantIntegerDef(root.getResult(0), 1))
      return mlir::failure();

    auto one = mlir::arith::ConstantOp::create(
        rewriter, root.getLoc(), rewriter.getIntegerAttr(resultType, 1));
    ++matchCount;
    return rewriter.replaceOp(root, one->getResults());
  }

private:
  unsigned &matchCount;
};

struct DemoMulByOneToAliasPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::arith::MulIOp> {
  explicit DemoMulByOneToAliasPattern(unsigned &matchCount)
      : matchCount(matchCount) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::arith::MulIOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::Value aliasedPayload;
    mlir::egraph::EValue aliasedValue;
    if (hasConstantIntegerDef(root.getOperand(0), 1)) {
      aliasedPayload = root.getOperation()->getOperand(1);
      aliasedValue = root.getOperand(1);
    } else if (hasConstantIntegerDef(root.getOperand(1), 1)) {
      aliasedPayload = root.getOperation()->getOperand(0);
      aliasedValue = root.getOperand(0);
    } else {
      return mlir::failure();
    }

    if (root.getResult(0).isEquivalentTo(aliasedValue))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 1> replacementValues = {aliasedPayload};
    ++matchCount;
    return rewriter.replaceOp(root, replacementValues);
  }

private:
  unsigned &matchCount;
};

mlir::FailureOr<mlir::egraph::EGraphExtractCost>
getDemoExtractCost(mlir::Operation *candidate) {
  llvm::StringRef operationName = candidate->getName().getStringRef();
  if (operationName == mlir::arith::ConstantOp::getOperationName())
    return mlir::egraph::EGraphExtractCost(1);
  if (operationName == mlir::arith::ShLIOp::getOperationName())
    return mlir::egraph::EGraphExtractCost(1);
  if (operationName == mlir::arith::MulIOp::getOperationName())
    return mlir::egraph::EGraphExtractCost(4);
  if (operationName == mlir::arith::DivSIOp::getOperationName())
    return mlir::egraph::EGraphExtractCost(8);
  return mlir::egraph::EGraphExtractCost(16);
}

mlir::LogicalResult runMatchAndExtractPipeline(mlir::ModuleOp module) {
  auto demoFunc = module.lookupSymbol<mlir::func::FuncOp>("arith_demo");
  if (!demoFunc)
    return module.emitError("expected a demo func.func named @arith_demo");

  unsigned mulByTwoMatches = 0;
  unsigned reassociateMatches = 0;
  unsigned divSelfMatches = 0;
  unsigned mulByOneMatches = 0;
  mlir::egraph::EGraphPatternSet patterns;
  patterns.add<DemoMulByTwoToShiftPattern>(mulByTwoMatches);
  patterns.add<DemoReassociateDivPattern>(reassociateMatches);
  patterns.add<DemoDivSelfToOnePattern>(divSelfMatches);
  patterns.add<DemoMulByOneToAliasPattern>(mulByOneMatches);

  mlir::Operation *demoOp = demoFunc.getOperation();
  if (mlir::failed(mlir::egraph::applyEGraphPatternsAndExtract(
          demoOp, patterns, mlir::egraph::EGraphExtractMode::Greedy,
          getDemoExtractCost, {}, /*recurseIntoNestedBlocks=*/false)))
    return module.emitError("egraph match/extract pipeline failed");

  if (mulByTwoMatches == 0 || reassociateMatches == 0 || divSelfMatches == 0 ||
      mulByOneMatches == 0)
    return module.emitError(
        "egraph match/extract did not exercise all demo patterns");

  mlir::Block &block = demoFunc.getBody().front();
  auto returnOp =
      llvm::dyn_cast_or_null<mlir::func::ReturnOp>(block.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1 ||
      returnOp.getOperand(0) != block.getArgument(0))
    return module.emitError(
        "egraph match/extract did not simplify the demo root to arg0");
  if (block.getOperations().size() != 1)
    return module.emitError(
        "egraph match/extract left redundant operations in the demo block");

  module.emitRemark() << "egraph pipeline matched and extracted arith demo";
  return mlir::success();
}

mlir::LogicalResult runRecursiveMatchAndExtractPipeline(mlir::ModuleOp module) {
  auto demoFunc = module.lookupSymbol<mlir::func::FuncOp>("nested_demo");
  if (!demoFunc)
    return module.emitError("expected a demo func.func named @nested_demo");

  unsigned mulByTwoMatches = 0;
  unsigned reassociateMatches = 0;
  unsigned divSelfMatches = 0;
  unsigned mulByOneMatches = 0;
  mlir::egraph::EGraphPatternSet patterns;
  patterns.add<DemoMulByTwoToShiftPattern>(mulByTwoMatches);
  patterns.add<DemoReassociateDivPattern>(reassociateMatches);
  patterns.add<DemoDivSelfToOnePattern>(divSelfMatches);
  patterns.add<DemoMulByOneToAliasPattern>(mulByOneMatches);

  mlir::Operation *demoOp = demoFunc.getOperation();
  mlir::egraph::EGraphMatchConfig config;
  if (mlir::failed(mlir::egraph::applyEGraphPatternsAndExtract(
          demoOp, patterns, mlir::egraph::EGraphExtractMode::Greedy,
          getDemoExtractCost, config, /*recurseIntoNestedBlocks=*/true)))
    return module.emitError("recursive egraph match/extract pipeline failed");

  if (mulByTwoMatches == 0 || reassociateMatches == 0 || divSelfMatches == 0 ||
      mulByOneMatches == 0)
    return module.emitError(
        "recursive egraph match/extract did not exercise all demo patterns");

  module.emitRemark()
      << "egraph recursive pipeline matched and extracted nested arith demo";
  return mlir::success();
}

struct TestEGraphMatchAndExtractPipelinePass
    : public mlir::PassWrapper<TestEGraphMatchAndExtractPipelinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestEGraphMatchAndExtractPipelinePass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-match-and-extract";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph match and extract pipeline";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::egraph::EGraphDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() final {
    if (mlir::failed(runMatchAndExtractPipeline(getOperation())))
      signalPassFailure();
  }
};

struct TestEGraphMatchAndExtractRecursivePipelinePass
    : public mlir::PassWrapper<TestEGraphMatchAndExtractRecursivePipelinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestEGraphMatchAndExtractRecursivePipelinePass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-match-and-extract-recursive";
  }

  llvm::StringRef getDescription() const final {
    return "test recursive MLIR-EGraph match and extract pipeline";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::egraph::EGraphDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    if (mlir::failed(runRecursiveMatchAndExtractPipeline(getOperation())))
      signalPassFailure();
  }
};

struct TestEGraphExtractInfoPass
    : public mlir::PassWrapper<TestEGraphExtractInfoPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEGraphExtractInfoPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-extract-info";
  }

  llvm::StringRef getDescription() const final {
    return "test normalized graph extract info selection";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::egraph::EGraphDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::egraph::EGraph graph;
    mlir::LogicalResult result = indexTestGraph(module, graph);
    if (mlir::succeeded(result))
      result = verifyExtractInfoSemantics(module, graph);

    if (mlir::failed(result))
      signalPassFailure();
  }
};

struct TestEGraphExtractCostModelPass
    : public mlir::PassWrapper<TestEGraphExtractCostModelPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEGraphExtractCostModelPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-extract-cost-model";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph single-node extract cost selection";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::egraph::EGraphDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::egraph::EGraph graph;
    mlir::LogicalResult result = indexTestGraph(module, graph);
    if (mlir::succeeded(result))
      result = verifyExtractCostModelSelection(module, graph);

    if (mlir::failed(result))
      signalPassFailure();
  }
};

struct TestEGraphGreedyExtractPass
    : public mlir::PassWrapper<TestEGraphGreedyExtractPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEGraphGreedyExtractPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-greedy-extract";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph greedy extract selection";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::egraph::EGraphDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::egraph::EGraph graph;
    mlir::LogicalResult result = indexTestGraph(module, graph);
    if (mlir::succeeded(result))
      result = verifyGreedyExtract(module, graph);

    if (mlir::failed(result))
      signalPassFailure();
  }
};

struct TestEGraphLinearProgrammingExtractPass
    : public mlir::PassWrapper<TestEGraphLinearProgrammingExtractPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestEGraphLinearProgrammingExtractPass)

  llvm::StringRef getArgument() const final { return "test-egraph-extract-lp"; }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph Z3-backed LP extract selection";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::egraph::EGraphDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::egraph::EGraph graph;
    mlir::LogicalResult result = indexTestGraph(module, graph);
    if (mlir::succeeded(result))
      result = verifyLinearProgrammingExtract(module, graph);

    if (mlir::failed(result))
      signalPassFailure();
  }
};

struct TestEGraphExtractMaterializationPass
    : public mlir::PassWrapper<TestEGraphExtractMaterializationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestEGraphExtractMaterializationPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-extract-materialization";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph extract materialization into ordinary MLIR";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::egraph::EGraphDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::egraph::EGraph graph;
    mlir::LogicalResult result = indexTestGraph(module, graph);
    if (mlir::succeeded(result))
      result = verifyGreedyExtractMaterialization(module, graph);

    if (mlir::failed(result))
      signalPassFailure();
  }
};

struct TestEGraphSymbolicEValuePass
    : public mlir::PassWrapper<TestEGraphSymbolicEValuePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEGraphSymbolicEValuePass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-symbolic-evalue";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph symbol-backed EValue lookup basics";
  }

  void runOnOperation() final {
    if (mlir::failed(verifySymbolicEValueLookup(getOperation())))
      signalPassFailure();
  }
};

struct TestEGraphSymbolicIndexPass
    : public mlir::PassWrapper<TestEGraphSymbolicIndexPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEGraphSymbolicIndexPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-symbolic-index";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph symbolic parent-use and structural indexes";
  }

  void runOnOperation() final {
    if (mlir::failed(verifySymbolicIndex(getOperation())))
      signalPassFailure();
  }
};

struct TestEGraphSymbolicUnionPass
    : public mlir::PassWrapper<TestEGraphSymbolicUnionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEGraphSymbolicUnionPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-symbolic-union";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph symbolic lazy union and persistent alias rebuild";
  }

  void runOnOperation() final {
    if (mlir::failed(verifySymbolicUnion(getOperation())))
      signalPassFailure();
  }
};

struct TestEGraphSymbolicFixedPointPass
    : public mlir::PassWrapper<TestEGraphSymbolicFixedPointPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEGraphSymbolicFixedPointPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-symbolic-fixed-point";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph symbolic rebuild fixed point";
  }

  void runOnOperation() final {
    if (mlir::failed(verifySymbolicFixedPoint(getOperation())))
      signalPassFailure();
  }
};

struct TestEGraphScratchRewriterPass
    : public mlir::PassWrapper<TestEGraphScratchRewriterPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEGraphScratchRewriterPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-scratch-rewriter";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph detached scratch block rewriter";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::egraph::EGraph graph;
    mlir::LogicalResult result = indexTestGraph(module, graph);
    if (mlir::failed(result)) {
      signalPassFailure();
      return;
    }

    unsigned initialRefCount = graph.getOpRefs().size();
    unsigned initialLeafCount = countLeafOps(module);
    bool checkedFailure = false;
    bool checkedNoEventSuccess = false;

    for (mlir::egraph::EOpRefBase ref : graph.getOpRefs()) {
      if (!checkedFailure &&
          llvm::isa<mlir::egraph::test::OpAOp>(ref.getOperation())) {
        llvm::SmallVector<mlir::Operation *> scratchOperations;
        ScratchFailurePattern pattern(scratchOperations);
        mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                           module.getContext());
        mlir::egraph::EGraphPatternRewriter rewriter(transaction);

        if (mlir::succeeded(pattern.matchAndRewrite(ref, rewriter)) ||
            scratchOperations.size() != 1)
          result = ref.getOperation()->emitOpError(
              "expected a failing pattern with one scratch op");
        else if (mlir::failed(verifyScratchOpBeforeDiscard(
                     graph, transaction, scratchOperations.front(),
                     ref.getOperation())))
          result = mlir::failure();
        else if (mlir::failed(verifyScratchDiscard(
                     graph, transaction, scratchOperations.front(),
                     ref.getOperation(),
                     "pattern failure discarded scratch block")))
          result = mlir::failure();
        else if (mlir::failed(verifyNoPersistentScratch(
                     module, graph, initialRefCount, initialLeafCount,
                     "pattern failure left persistent egraph unchanged")))
          result = mlir::failure();

        checkedFailure = true;
      }

      if (!checkedNoEventSuccess &&
          llvm::isa<mlir::egraph::test::OpBOp>(ref.getOperation())) {
        llvm::SmallVector<mlir::Operation *> scratchOperations;
        ScratchNoEventPattern pattern(scratchOperations);
        mlir::egraph::EGraphRewriteTransaction transaction(graph,
                                                           module.getContext());
        mlir::egraph::EGraphPatternRewriter rewriter(transaction);

        if (mlir::failed(pattern.matchAndRewrite(ref, rewriter)) ||
            scratchOperations.size() != 1)
          result = ref.getOperation()->emitOpError(
              "expected a successful no-event pattern with one scratch op");
        else if (transaction.hasEvents())
          result = ref.getOperation()->emitOpError(
              "scratch-only transaction unexpectedly recorded events");
        else if (mlir::failed(verifyScratchOpBeforeDiscard(
                     graph, transaction, scratchOperations.front(),
                     ref.getOperation())))
          result = mlir::failure();
        else if (mlir::failed(verifyScratchDiscard(
                     graph, transaction, scratchOperations.front(),
                     ref.getOperation(),
                     "success without events discarded scratch block")))
          result = mlir::failure();
        else if (mlir::failed(verifyNoPersistentScratch(
                     module, graph, initialRefCount, initialLeafCount,
                     "success without events left persistent egraph "
                     "unchanged")))
          result = mlir::failure();

        checkedNoEventSuccess = true;
      }
    }

    if (!checkedFailure)
      result = module.emitError("expected an op_a candidate");
    if (!checkedNoEventSuccess)
      result = module.emitError("expected an op_b candidate");

    if (mlir::failed(result))
      signalPassFailure();
  }
};

struct TestEGraphRewriteEventsPass
    : public mlir::PassWrapper<TestEGraphRewriteEventsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEGraphRewriteEventsPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-rewrite-events";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph rewrite event recording";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::egraph::EGraph graph;
    mlir::LogicalResult result = indexTestGraph(module, graph);
    if (mlir::succeeded(result))
      result = verifyRewriteEventRecording(module, graph);

    if (mlir::failed(result))
      signalPassFailure();
  }
};

struct TestEGraphEventValidationPass
    : public mlir::PassWrapper<TestEGraphEventValidationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEGraphEventValidationPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-event-validation";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph transaction event validation";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::egraph::EGraph graph;
    mlir::LogicalResult result = indexTestGraph(module, graph);
    if (mlir::succeeded(result))
      result = verifyRewriteEventValidation(module, graph);

    if (mlir::failed(result))
      signalPassFailure();
  }
};

struct TestEGraphNegativeFailuresPass
    : public mlir::PassWrapper<TestEGraphNegativeFailuresPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEGraphNegativeFailuresPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-negative-failures";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph negative failure diagnostics";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::egraph::EGraph graph;
    mlir::LogicalResult result = indexTestGraph(module, graph);
    if (mlir::succeeded(result))
      result = verifyNegativeFailures(module, graph);

    if (mlir::failed(result))
      signalPassFailure();
  }
};

struct TestEGraphSymbolicTransactionCommitPass
    : public mlir::PassWrapper<TestEGraphSymbolicTransactionCommitPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestEGraphSymbolicTransactionCommitPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-symbolic-transaction-commit";
  }

  llvm::StringRef getDescription() const final {
    return "test symbolic MLIR-EGraph scratch DAG commit";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::egraph::EGraph graph;
    mlir::LogicalResult result = indexTestGraph(module, graph);
    if (mlir::succeeded(result))
      result = verifySymbolicTransactionCommit(module, graph);

    if (mlir::failed(result))
      signalPassFailure();
  }
};

struct TestEGraphWorklistDriverPass
    : public mlir::PassWrapper<TestEGraphWorklistDriverPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEGraphWorklistDriverPass)

  llvm::StringRef getArgument() const final {
    return "test-egraph-worklist-driver";
  }

  llvm::StringRef getDescription() const final {
    return "test MLIR-EGraph worklist-driven match driver";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::egraph::EGraph graph;
    mlir::LogicalResult result = indexTestGraph(module, graph);
    if (mlir::succeeded(result)) {
      bool hasSymbolicGraph =
          llvm::any_of(module.getOps<mlir::egraph::EGraphOp>(),
                       [](mlir::egraph::EGraphOp) { return true; });
      result = hasSymbolicGraph ? verifySymbolicWorklistDriver(module, graph)
                                : verifyWorklistDriver(module, graph);
    }

    if (mlir::failed(result))
      signalPassFailure();
  }
};

} // namespace

void registerEGraphTestPasses() {
  mlir::PassRegistration<TestEGraphMatchAndExtractPipelinePass>();
  mlir::PassRegistration<TestEGraphMatchAndExtractRecursivePipelinePass>();
  mlir::PassRegistration<TestEGraphExtractInfoPass>();
  mlir::PassRegistration<TestEGraphExtractCostModelPass>();
  mlir::PassRegistration<TestEGraphGreedyExtractPass>();
  mlir::PassRegistration<TestEGraphLinearProgrammingExtractPass>();
  mlir::PassRegistration<TestEGraphExtractMaterializationPass>();
  mlir::PassRegistration<TestEGraphSymbolicEValuePass>();
  mlir::PassRegistration<TestEGraphSymbolicIndexPass>();
  mlir::PassRegistration<TestEGraphSymbolicUnionPass>();
  mlir::PassRegistration<TestEGraphSymbolicFixedPointPass>();
  mlir::PassRegistration<TestEGraphScratchRewriterPass>();
  mlir::PassRegistration<TestEGraphRewriteEventsPass>();
  mlir::PassRegistration<TestEGraphEventValidationPass>();
  mlir::PassRegistration<TestEGraphNegativeFailuresPass>();
  mlir::PassRegistration<TestEGraphSymbolicTransactionCommitPass>();
  mlir::PassRegistration<TestEGraphWorklistDriverPass>();
}
#endif
