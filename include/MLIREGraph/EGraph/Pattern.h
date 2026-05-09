#ifndef MLIR_EGRAPH_EGRAPH_PATTERN_H
#define MLIR_EGRAPH_EGRAPH_PATTERN_H

#include "MLIREGraph/EGraph/EGraph.h"
#include "MLIREGraph/EGraph/Extract.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseMap.h"
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
};

/// Match driver knobs. Zero-valued limits mean unlimited.
struct EGraphMatchConfig {
  unsigned maxIterations = 0;
  unsigned maxCandidatesPerEClass = 0;
  /// Treat MLIR fold results as egraph rewrite alternatives, not in-place IR
  /// canonicalization.
  bool enableMlirFold = false;
};

/// Summary of a match run.
struct EGraphMatchStats {
  bool changed = false;
  bool limitReached = false;
  EGraphMatchLimit reachedLimit = EGraphMatchLimit::None;
  unsigned iterations = 0;
  unsigned enqueuedCandidates = 0;
  unsigned skippedCandidateCap = 0;
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

enum class EGraphPatternRootKind {
  Any,
  OperationName,
  TraitID,
  InterfaceID,
};

class EGraphPattern {
public:
  virtual ~EGraphPattern() = default;

  /// Returns how this pattern chooses root operations.
  virtual EGraphPatternRootKind getRootKind() const = 0;
  /// Returns the root operation name for operation-name rooted patterns.
  virtual StringRef getRootOperationName() const { return {}; }
  /// Returns the root trait ID for trait-rooted patterns.
  virtual TypeID getRootTraitID() const { return TypeID(); }
  /// Returns the root interface ID for interface-rooted patterns.
  virtual TypeID getRootInterfaceID() const { return TypeID(); }
  /// Matches and rewrites a live egraph operation occurrence.
  virtual LogicalResult
  matchAndRewrite(EOpRefBase root, EGraphPatternRewriter &rewriter) const = 0;
};

class EGraphAnyOpPattern : public EGraphPattern {
public:
  /// Matches any operation type. The concrete pattern performs final checks.
  EGraphPatternRootKind getRootKind() const final {
    return EGraphPatternRootKind::Any;
  }
};

template <typename OpTy>
class EGraphPatternFor : public EGraphPattern {
public:
  /// Matches only the requested operation type.
  EGraphPatternRootKind getRootKind() const final {
    return EGraphPatternRootKind::OperationName;
  }

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

template <template <typename> class TraitTy>
class EGraphTraitPattern : public EGraphPattern {
public:
  /// Matches operations registered with the requested trait.
  EGraphPatternRootKind getRootKind() const final {
    return EGraphPatternRootKind::TraitID;
  }

  /// Returns the TypeID of the requested root trait.
  TypeID getRootTraitID() const final { return TypeID::get<TraitTy>(); }

  /// Rechecks the trait before invoking the concrete pattern.
  LogicalResult matchAndRewrite(EOpRefBase root,
                                EGraphPatternRewriter &rewriter) const final {
    if (!root || !root.getOperation()->hasTrait<TraitTy>())
      return failure();
    return matchTraitRoot(root, rewriter);
  }

  /// Rewrites a live root occurrence with the requested trait.
  virtual LogicalResult
  matchTraitRoot(EOpRefBase root, EGraphPatternRewriter &rewriter) const = 0;
};

template <typename InterfaceTy>
class EGraphInterfacePattern : public EGraphPattern {
public:
  /// Matches operations implementing the requested interface.
  EGraphPatternRootKind getRootKind() const final {
    return EGraphPatternRootKind::InterfaceID;
  }

  /// Returns the TypeID of the requested root interface.
  TypeID getRootInterfaceID() const final { return InterfaceTy::getInterfaceID(); }

  /// Rechecks the interface before invoking the concrete pattern.
  LogicalResult matchAndRewrite(EOpRefBase root,
                                EGraphPatternRewriter &rewriter) const final {
    if (!root ||
        !root.getOperation()->getName().hasInterface(getRootInterfaceID()))
      return failure();
    return matchInterfaceRoot(root, rewriter);
  }

  /// Rewrites a live root occurrence implementing the requested interface.
  virtual LogicalResult
  matchInterfaceRoot(EOpRefBase root, EGraphPatternRewriter &rewriter) const = 0;
};

class EGraphPatternSet {
public:
  /// Adds an owned pattern and indexes it by root dispatch kind.
  void add(std::unique_ptr<EGraphPattern> pattern) {
    EGraphPattern *rawPattern = pattern.get();
    switch (rawPattern->getRootKind()) {
    case EGraphPatternRootKind::Any:
      anyRootPatterns.push_back(rawPattern);
      break;
    case EGraphPatternRootKind::OperationName:
      patternsByRoot[rawPattern->getRootOperationName()].push_back(rawPattern);
      break;
    case EGraphPatternRootKind::TraitID:
      patternsByTrait[rawPattern->getRootTraitID()].push_back(rawPattern);
      break;
    case EGraphPatternRootKind::InterfaceID:
      patternsByInterface[rawPattern->getRootInterfaceID()].push_back(
          rawPattern);
      break;
    }
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

  /// Returns exact operation-name rooted patterns for the given root name.
  ArrayRef<EGraphPattern *> lookup(StringRef operationName) const {
    auto it = patternsByRoot.find(operationName);
    if (it == patternsByRoot.end())
      return {};
    return it->second;
  }

  /// Appends patterns that may match the given operation.
  void lookup(Operation *operation,
              SmallVectorImpl<EGraphPattern *> &result) const {
    if (!operation)
      return;

    ArrayRef<EGraphPattern *> exactPatterns =
        lookup(operation->getName().getStringRef());
    result.append(exactPatterns.begin(), exactPatterns.end());

    for (const auto &it : patternsByTrait)
      if (operation->getName().hasTrait(it.first))
        result.append(it.second.begin(), it.second.end());

    for (const auto &it : patternsByInterface)
      if (operation->getName().hasInterface(it.first))
        result.append(it.second.begin(), it.second.end());

    result.append(anyRootPatterns.begin(), anyRootPatterns.end());
  }

private:
  std::vector<std::unique_ptr<EGraphPattern>> ownedPatterns;
  SmallVector<EGraphPattern *> anyRootPatterns;
  llvm::StringMap<SmallVector<EGraphPattern *>> patternsByRoot;
  llvm::DenseMap<TypeID, SmallVector<EGraphPattern *>> patternsByTrait;
  llvm::DenseMap<TypeID, SmallVector<EGraphPattern *>> patternsByInterface;
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
