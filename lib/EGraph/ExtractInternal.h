#ifndef MLIR_EGRAPH_LIB_EGRAPH_EXTRACTINTERNAL_H
#define MLIR_EGRAPH_LIB_EGRAPH_EXTRACTINTERNAL_H

#include "MLIREGraph/EGraph/Extract.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

namespace mlir::egraph::detail {

struct EGraphExtractRequest {
  EGraphExtractMode mode = EGraphExtractMode::Greedy;
  SmallVector<EValue, 4> roots;
};

struct ExtractSelection {
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

inline FailureOr<SmallVector<FlatSymbolRefAttr, 4>>
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

inline FailureOr<StringAttr>
getAliasCandidateTarget(EClassOp eclass, unsigned candidateOrdinal) {
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

inline bool isAliasCandidate(EClassOp eclass, unsigned candidateOrdinal) {
  return succeeded(getAliasCandidateTarget(eclass, candidateOrdinal));
}

inline FailureOr<EGraphExtractCost> addExtractCosts(EGraphExtractCost lhs,
                                                    EGraphExtractCost rhs) {
  if (rhs > std::numeric_limits<EGraphExtractCost>::max() - lhs)
    return failure();
  return lhs + rhs;
}

inline FailureOr<EOpRefBase> getCandidateRootRef(EGraph &graph, EClassOp eclass,
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

FailureOr<EGraphExtractInfo>
runGreedyExtract(EGraph &graph, EGraphOp egraph,
                 const EGraphExtractRequest &request,
                 EGraphExtractCostModel costModel);

FailureOr<EGraphExtractInfo>
runZ3LinearProgrammingExtract(EGraph &graph, EGraphOp egraph,
                              const EGraphExtractRequest &request,
                              EGraphExtractCostModel costModel);

FailureOr<EGraphExtractInfo>
runOrToolsLinearProgrammingExtract(EGraph &graph, EGraphOp egraph,
                                   const EGraphExtractRequest &request,
                                   EGraphExtractCostModel costModel);

} // namespace mlir::egraph::detail

#endif // MLIR_EGRAPH_LIB_EGRAPH_EXTRACTINTERNAL_H
