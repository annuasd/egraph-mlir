#ifndef MLIR_EGRAPH_EGRAPH_EXTRACT_H
#define MLIR_EGRAPH_EGRAPH_EXTRACT_H

#include "MLIREGraph/EGraph/EGraph.h"

namespace mlir {
namespace egraph {

class GraphMatchState;

enum class EGraphExtractMode {
  Greedy,
  LinearProgramming,
};

/// Scalar cost assigned to a single candidate occurrence.
using EGraphExtractCost = uint64_t;

/// Cost model for a single candidate operation.
using EGraphExtractCostModel =
    llvm::function_ref<FailureOr<EGraphExtractCost>(Operation *)>;

/// Candidate selection summary produced by an extraction backend.
struct EGraphExtractInfo {
  EGraphExtractMode mode = EGraphExtractMode::Greedy;
  /// Ordered by egraph result slot.
  SmallVector<EValue, 4> roots;
  /// Total extraction cost for `roots[i]`.
  SmallVector<EGraphExtractCost, 4> rootCosts;
  /// Backend-specific order of e-classes assigned a concrete candidate.
  SmallVector<EValue, 4> selectedEClasses;
  /// `selectedCandidateRoots[i]` corresponds to `selectedEClasses[i]`.
  SmallVector<EOpRefBase, 4> selectedCandidateRoots;
  /// Local cost reported by the cost model for `selectedCandidateRoots[i]`.
  SmallVector<EGraphExtractCost, 4> selectedCandidateCosts;
  /// Total subtree cost for `selectedCandidateRoots[i]`.
  SmallVector<EGraphExtractCost, 4> selectedSubtreeCosts;
  /// Alias e-classes selected during extraction.
  SmallVector<EValue, 4> selectedAliasEClasses;
  /// `selectedAliasTargets[i]` corresponds to `selectedAliasEClasses[i]`.
  SmallVector<EValue, 4> selectedAliasTargets;
  bool changed = false;
};

/// Formats an extraction mode for diagnostics and debug output.
StringRef stringifyEGraphExtractMode(EGraphExtractMode mode);

/// Updates the original block in place by extracting from the rewritten state.
/// Explicit roots default to the enclosing egraph return targets.
LogicalResult extractEGraph(GraphMatchState &state, EGraphExtractMode mode,
                            EGraphExtractCostModel costModel,
                            ArrayRef<EValue> explicitRoots = {},
                            EGraphExtractInfo *info = nullptr);

/// Extracts from a symbolic egraph container and returns the selected
/// candidates without materializing them into a block.
LogicalResult extractEGraph(EGraph &graph, EGraphOp egraph,
                            EGraphExtractMode mode,
                            EGraphExtractCostModel costModel,
                            EGraphExtractInfo *info = nullptr,
                            ArrayRef<EValue> explicitRoots = {});

/// Materializes an extraction selection into ordinary MLIR values.
FailureOr<SmallVector<Value, 4>>
materializeEGraphExtractInfo(EGraph &graph, EGraphOp egraph,
                             const EGraphExtractInfo &selection,
                             OpBuilder &builder, ArrayRef<Value> inputValues);

} // namespace egraph
} // namespace mlir

#endif // MLIR_EGRAPH_EGRAPH_EXTRACT_H
