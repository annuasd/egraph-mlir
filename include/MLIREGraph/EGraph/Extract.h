#ifndef MLIR_EGRAPH_EGRAPH_EXTRACT_H
#define MLIR_EGRAPH_EGRAPH_EXTRACT_H

#include "MLIREGraph/EGraph/EGraph.h"

namespace mlir {
namespace egraph {

enum class EGraphExtractMode {
  Greedy,
  LinearProgramming,
};

/// Scalar cost assigned to a single candidate occurrence.
using EGraphExtractCost = uint64_t;

/// Cost model for a single candidate occurrence.
using EGraphExtractCostModel =
    llvm::function_ref<FailureOr<EGraphExtractCost>(EOpRefBase)>;

/// Normalized extractor input shared by all extraction backends.
struct EGraphExtractRequest {
  EGraphExtractMode mode = EGraphExtractMode::Greedy;
  /// Ordered by egraph result slot.
  SmallVector<EValue, 4> roots;
};

/// Candidate selection summary produced by an extraction backend.
struct EGraphExtractResult {
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

/// Builds a normalized extract request from explicit roots, or from the
/// enclosing `egraph.return` targets when explicit roots are empty.
///
/// Extraction is a post-saturation phase, so request construction rejects dirty
/// graphs. Returned roots are leader-normalized, symbol-backed EValues ordered
/// by egraph result slot.
FailureOr<EGraphExtractRequest>
buildEGraphExtractRequest(EGraph &graph, EGraphOp egraph,
                          ArrayRef<EValue> explicitRoots = {},
                          EGraphExtractMode mode = EGraphExtractMode::Greedy);

/// Selects the lowest-cost candidate root for each requested result slot.
/// The graph must already be clean, and the request roots must already be
/// normalized.
FailureOr<EGraphExtractResult>
selectEGraphExtractCandidates(EGraph &graph,
                              const EGraphExtractRequest &request,
                              EGraphExtractCostModel costModel);

/// Extracts a greedy candidate selection for the requested result graph.
/// The graph must already be clean, and the request must use greedy mode.
FailureOr<EGraphExtractResult>
extractEGraphGreedily(EGraph &graph, EGraphOp egraph,
                      const EGraphExtractRequest &request,
                      EGraphExtractCostModel costModel);

/// Extracts a globally optimal candidate selection for the requested result
/// graph using the Z3-backed linear programming backend.
/// The graph must already be clean, and the request must use LP mode.
FailureOr<EGraphExtractResult>
extractEGraphByLinearProgramming(EGraph &graph, EGraphOp egraph,
                                 const EGraphExtractRequest &request,
                                 EGraphExtractCostModel costModel);

/// Materializes the selected extraction result into ordinary MLIR values.
/// `inputValues` are ordered like the enclosing egraph entry arguments.
/// Shared selected e-classes are cloned once and reused by all roots.
FailureOr<SmallVector<Value, 4>>
materializeEGraphExtractResult(EGraph &graph, EGraphOp egraph,
                               const EGraphExtractResult &selection,
                               OpBuilder &builder, ArrayRef<Value> inputValues);

} // namespace egraph
} // namespace mlir

#endif // MLIR_EGRAPH_EGRAPH_EXTRACT_H
