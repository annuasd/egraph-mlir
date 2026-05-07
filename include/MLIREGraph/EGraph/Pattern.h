#ifndef MLIR_EGRAPH_EGRAPH_PATTERN_H
#define MLIR_EGRAPH_EGRAPH_PATTERN_H

#include "MLIREGraph/EGraph/EGraph.h"
#include "MLIREGraph/EGraph/Extract.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlir {
namespace egraph {

class EGraphPatternSet;

enum class EGraphRewriteEventKind {
  /// Records a replacement event for the root candidate.
  ReplaceOp,
  /// Records an equivalence event between two candidates.
  ReplaceAllUsesWith,
};

struct EGraphRewriteEvent {
  /// The kind of rewrite event recorded by the transaction.
  EGraphRewriteEventKind kind = EGraphRewriteEventKind::ReplaceOp;
  /// The rewritten candidate root.
  EOpRefBase root;
  /// Replacement values recorded for the event.
  SmallVector<Value, 4> replacementValues;
};

struct EGraphInternedValue {
  /// The original scratch value.
  Value originalValue;
  /// The persistent egraph value created for the scratch value.
  EValue value;
};

struct EGraphTransactionInternResult {
  /// Scratch operations reachable from the rewrite events.
  SmallVector<EGraphInternedOperation, 4> operations;
  /// Event replacement values mapped to persistent egraph values.
  SmallVector<EGraphInternedValue, 4> eventValues;
};

struct EGraphRewriteCommitResult {
  /// True if the commit changed the graph.
  bool changed = false;
  /// Number of rebuild rounds performed by the commit.
  unsigned rebuilds = 0;
  /// Roots of new candidates discovered during commit.
  SmallVector<EOpRefBase, 4> newCandidateRoots;
  /// Parent candidates affected by the commit.
  SmallVector<EOpRefBase, 4> affectedParentCandidates;
};

enum class EGraphMatchLimit {
  None,
  Iteration,
  Candidate,
  Rebuild,
};

/// Match limits and tracing knobs. Zero-valued limits mean unlimited.
struct EGraphMatchConfig {
  unsigned maxIterations = 0;
  unsigned maxEnqueuedCandidates = 0;
  unsigned maxRebuilds = 0;
  /// Attempt MLIR fold on each dispatched candidate before user rewrites.
  bool enableMlirFold = false;
  /// Optional debug stream for driver tracing.
  llvm::raw_ostream *debugStream = nullptr;
};

/// Summary of a match run.
struct EGraphMatchStats {
  bool changed = false;
  bool limitReached = false;
  EGraphMatchLimit reachedLimit = EGraphMatchLimit::None;
  unsigned iterations = 0;
  unsigned enqueuedCandidates = 0;
  unsigned skippedStaleRefs = 0;
  unsigned matchedPatterns = 0;
  unsigned changedCommits = 0;
  unsigned rebuilds = 0;
};

/// Move-only state produced by applying match patterns to a block.
class GraphMatchState {
public:
  GraphMatchState() = delete;
  GraphMatchState(GraphMatchState &&) = default;
  GraphMatchState &operator=(GraphMatchState &&) = default;
  GraphMatchState(const GraphMatchState &) = delete;
  GraphMatchState &operator=(const GraphMatchState &) = delete;

  Block &getBlock() const;
  const EGraphMatchStats &getStats() const;

private:
  friend FailureOr<GraphMatchState>
  applyEGraphPatterns(Block &block, const EGraphPatternSet &patterns,
                      const EGraphMatchConfig &config);
  friend LogicalResult extractEGraph(GraphMatchState &state,
                                     EGraphExtractMode mode,
                                     EGraphExtractCostModel costModel,
                                     ArrayRef<EValue> explicitRoots,
                                     EGraphExtractInfo *info);
  friend LogicalResult applyEGraphPatternsAndExtract(
      Block &block, const EGraphPatternSet &patterns, EGraphExtractMode mode,
      EGraphExtractCostModel costModel, const EGraphMatchConfig &config);

  GraphMatchState(Block &block, std::unique_ptr<EGraph> graph,
                  OwningOpRef<Operation *> egraphOp, EGraphMatchStats stats);

  Block *block = nullptr;
  std::unique_ptr<EGraph> graph;
  OwningOpRef<Operation *> egraphOp;
  EGraphMatchStats stats;
  bool extracted = false;
};

class EGraphRewriteTransaction {
public:
  EGraphRewriteTransaction(EGraph &graph, MLIRContext *context);
  EGraphRewriteTransaction(const EGraphRewriteTransaction &) = delete;
  EGraphRewriteTransaction &
  operator=(const EGraphRewriteTransaction &) = delete;
  ~EGraphRewriteTransaction();

  EGraph &getGraph() const { return *graph; }
  MLIRContext *getContext() const { return context; }
  Block *getScratchBlock() const { return scratchBlock.get(); }
  bool hasEvents() const { return !rewriteEvents.empty(); }
  bool isDiscarded() const { return discarded; }
  ArrayRef<EGraphRewriteEvent> getRewriteEvents() const {
    return rewriteEvents;
  }

  /// Validates that all recorded events belong to the active transaction.
  LogicalResult validateEvents() const;
  /// Collects the scratch DAG reachable from the recorded rewrite events.
  FailureOr<SmallVector<Operation *>> collectReachableScratchOperations() const;
  /// Interns the reachable scratch DAG into persistent egraph values.
  FailureOr<EGraphTransactionInternResult> internReachableScratchOperations();
  /// Unions interned values according to the recorded rewrite events.
  FailureOr<EGraphUnionResult>
  unionInternedEventValues(const EGraphTransactionInternResult &interned);
  /// Commits the transaction into the graph, optionally rebuilding immediately.
  FailureOr<EGraphRewriteCommitResult> commit(Operation *rebuildRoot,
                                              bool rebuildNow = true);

  /// Discards the transaction and drops all recorded events.
  void discard();

private:
  class ScratchListener;
  friend class EGraphPatternRewriter;
  friend class ScratchListener;

  OpBuilder::Listener *getListener() const;
  void notifyScratchOperationInserted(Operation *operation);
  /// Records a rewrite event for the given root and replacement values.
  LogicalResult recordRewriteEvent(EGraphRewriteEventKind kind, EOpRefBase root,
                                   ValueRange replacementValues);
  /// Records an equivalence between two operation occurrences.
  LogicalResult recordEquivalence(EOpRefBase fromOp, Operation *toOperation,
                                  EOpRefBase toOp = EOpRefBase());
  LogicalResult
  collectReachableScratchOperation(Operation *operation,
                                   llvm::SmallPtrSetImpl<Operation *> &visited,
                                   SmallVectorImpl<Operation *> &ordered) const;
  LogicalResult
  collectReachableScratchValue(Value value,
                               llvm::SmallPtrSetImpl<Operation *> &visited,
                               SmallVectorImpl<Operation *> &ordered) const;
  bool ownsScratchOperation(Operation *operation) const;
  bool ownsScratchValue(Value value) const;

  EGraph *graph = nullptr;
  MLIRContext *context = nullptr;
  std::unique_ptr<Block> scratchBlock;
  std::unique_ptr<ScratchListener> listener;
  SmallVector<Operation *> scratchOperations;
  SmallVector<EGraphRewriteEvent> rewriteEvents;
  bool discarded = false;
};

class EGraphPatternRewriter : public OpBuilder {
public:
  explicit EGraphPatternRewriter(EGraphRewriteTransaction &transaction);

  EGraphPatternRewriter(const EGraphPatternRewriter &) = delete;
  EGraphPatternRewriter &operator=(const EGraphPatternRewriter &) = delete;
  virtual ~EGraphPatternRewriter() = default;

  EGraphRewriteTransaction &getTransaction() const { return *transaction; }
  Block *getScratchBlock() const { return transaction->getScratchBlock(); }

  /// Records a replacement event for the given root candidate.
  LogicalResult replaceOp(EOpRefBase oldOp, ValueRange replacementValues);
  /// Records a replacement event for the given operation occurrence.
  LogicalResult replaceOp(Operation *oldOp, ValueRange replacementValues);

  /// Records an equivalence event between two operation occurrences.
  LogicalResult replaceAllUsesWith(EOpRefBase fromOp, EOpRefBase toOp);
  /// Records an equivalence event between two operations.
  LogicalResult replaceAllUsesWith(Operation *fromOp, Operation *toOp);

  void clearInsertionPoint() = delete;
  void restoreInsertionPoint(OpBuilder::InsertPoint ip) = delete;
  void setInsertionPoint(Block *block, Block::iterator insertPoint) = delete;
  void setInsertionPoint(Operation *op) = delete;
  void setInsertionPointAfter(Operation *op) = delete;
  void setInsertionPointAfterValue(Value val) = delete;
  void setInsertionPointToStart(Block *block) = delete;
  void setInsertionPointToEnd(Block *block) = delete;

private:
  EGraphRewriteTransaction *transaction = nullptr;
};

class EGraphPattern {
public:
  virtual ~EGraphPattern() = default;

  /// Returns the root operation name this pattern matches.
  virtual StringRef getRootOperationName() const = 0;
  /// Matches and rewrites a live egraph operation occurrence.
  virtual LogicalResult
  matchAndRewrite(EOpRefBase root, EGraphPatternRewriter &rewriter) const = 0;
};

template <typename OpTy>
class EGraphPatternFor : public EGraphPattern {
public:
  /// Returns the operation name of the wrapped root operation type.
  StringRef getRootOperationName() const final {
    return OpTy::getOperationName();
  }

  /// Downcasts the root occurrence to the requested operation type.
  LogicalResult matchAndRewrite(EOpRefBase root,
                                EGraphPatternRewriter &rewriter) const final {
    if (!root || !llvm::isa<OpTy>(root.getOperation()))
      return failure();
    return matchAndRewrite(EOpRef<OpTy>(root), rewriter);
  }

  /// Rewrites a live root occurrence of the requested operation type.
  virtual LogicalResult
  matchAndRewrite(EOpRef<OpTy> root, EGraphPatternRewriter &rewriter) const = 0;
};

class EGraphPatternSet {
public:
  /// Adds an owned pattern to the set and indexes it by root operation name.
  void add(std::unique_ptr<EGraphPattern> pattern) {
    EGraphPattern *rawPattern = pattern.get();
    patternsByRoot[rawPattern->getRootOperationName()].push_back(rawPattern);
    ownedPatterns.push_back(std::move(pattern));
  }

  template <typename PatternTy, typename... Args>
  PatternTy &add(Args &&...args) {
    static_assert(std::is_base_of_v<EGraphPattern, PatternTy>,
                  "pattern must derive from EGraphPattern");
    auto pattern = std::make_unique<PatternTy>(std::forward<Args>(args)...);
    PatternTy &result = *pattern;
    add(std::move(pattern));
    return result;
  }

  /// Returns the patterns registered for the given root operation name.
  ArrayRef<EGraphPattern *> lookup(StringRef operationName) const {
    auto it = patternsByRoot.find(operationName);
    if (it == patternsByRoot.end())
      return {};
    return it->second;
  }

private:
  std::vector<std::unique_ptr<EGraphPattern>> ownedPatterns;
  llvm::StringMap<SmallVector<EGraphPattern *>> patternsByRoot;
};

/// Applies egraph patterns to exactly one block.
FailureOr<GraphMatchState>
applyEGraphPatterns(Block &block, const EGraphPatternSet &patterns,
                    const EGraphMatchConfig &config = {});
/// Applies egraph patterns to blocks owned by an operation. By default only
/// direct region blocks are processed. Recursive mode skips container blocks
/// with nested-region ops and processes nested flat blocks instead.
FailureOr<EGraphMatchStats>
applyEGraphPatterns(Operation *op, const EGraphPatternSet &patterns,
                    const EGraphMatchConfig &config = {},
                    bool recurseIntoNestedBlocks = false);

/// Applies egraph patterns to exactly one block and extracts the result.
LogicalResult applyEGraphPatternsAndExtract(
    Block &block, const EGraphPatternSet &patterns, EGraphExtractMode mode,
    EGraphExtractCostModel costModel, const EGraphMatchConfig &config = {});
/// Applies egraph patterns to operation-owned blocks and extracts each result.
LogicalResult applyEGraphPatternsAndExtract(
    Operation *op, const EGraphPatternSet &patterns, EGraphExtractMode mode,
    EGraphExtractCostModel costModel, const EGraphMatchConfig &config = {},
    bool recurseIntoNestedBlocks = false);

/// Formats a match limit for diagnostics.
StringRef stringifyEGraphMatchLimit(EGraphMatchLimit limit);
/// Prints match stats to the given stream.
void printEGraphMatchStats(llvm::raw_ostream &os,
                           const EGraphMatchStats &stats);

} // namespace egraph
} // namespace mlir

#endif // MLIR_EGRAPH_EGRAPH_PATTERN_H
