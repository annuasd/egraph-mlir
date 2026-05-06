#ifndef MLIR_EGRAPH_EGRAPH_EGRAPH_H
#define MLIR_EGRAPH_EGRAPH_EGRAPH_H

#include "MLIREGraph/IR/EGraphOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cstdint>
#include <type_traits>
#include <utility>

namespace mlir {
namespace egraph {

class EGraph;
class EValue;
class EOpRefBase;
class EGraphRewriteTransaction;
template <typename OpTy>
class EOpRef;

using EOpOccurrenceId = uint64_t;

enum class OperationOwnership {
  EGraphOwned,
  ScratchCreated,
  IllegalExternal,
};

/// Structural key used to hash-cons a candidate during rebuild.
struct EGraphStructuralKey {
  OperationName operationName = OperationName::getFromOpaquePointer(nullptr);
  DictionaryAttr attributes;
  SmallVector<Type, 4> resultTypes;
  SmallVector<Value, 4> childKeyValues;
  SmallVector<StringAttr, 4> childLeaderSymbols;
};

inline bool operator==(const EGraphStructuralKey &lhs,
                       const EGraphStructuralKey &rhs) {
  return lhs.operationName == rhs.operationName &&
         lhs.attributes == rhs.attributes &&
         lhs.resultTypes == rhs.resultTypes &&
         lhs.childKeyValues == rhs.childKeyValues &&
         lhs.childLeaderSymbols == rhs.childLeaderSymbols;
}

inline bool operator!=(const EGraphStructuralKey &lhs,
                       const EGraphStructuralKey &rhs) {
  return !(lhs == rhs);
}

/// Hash-cons payload for an interned candidate.
struct EGraphHashConsEntry {
  EClassOp eclass;
  Operation *operation = nullptr;
  SmallVector<Value, 4> resultKeyValues;
  SmallVector<StringAttr, 4> resultSymbols;
};

} // namespace egraph
} // namespace mlir

namespace llvm {
template <>
struct DenseMapInfo<mlir::egraph::EGraphStructuralKey> {
  static mlir::egraph::EGraphStructuralKey getEmptyKey() {
    mlir::egraph::EGraphStructuralKey key;
    key.operationName = DenseMapInfo<mlir::OperationName>::getEmptyKey();
    return key;
  }

  static mlir::egraph::EGraphStructuralKey getTombstoneKey() {
    mlir::egraph::EGraphStructuralKey key;
    key.operationName = DenseMapInfo<mlir::OperationName>::getTombstoneKey();
    return key;
  }

  static unsigned getHashValue(const mlir::egraph::EGraphStructuralKey &key) {
    return static_cast<unsigned>(
        llvm::hash_combine(key.operationName, key.attributes,
                           llvm::hash_combine_range(key.resultTypes),
                           llvm::hash_combine_range(key.childKeyValues),
                           llvm::hash_combine_range(key.childLeaderSymbols)));
  }

  static bool isEqual(const mlir::egraph::EGraphStructuralKey &lhs,
                      const mlir::egraph::EGraphStructuralKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace mlir {
namespace egraph {

/// Stable handle for a live candidate occurrence in the indexed EGraph.
class EOpRefBase {
public:
  EOpRefBase() = default;
  EOpRefBase(EGraph *graph, EOpOccurrenceId id, unsigned generation)
      : graph(graph), id(id), generation(generation) {}

  EGraph *getGraph() const { return graph; }

  explicit operator bool() const { return graph != nullptr; }

  bool isLive() const;
  bool isSameCandidateAs(EOpRefBase other) const;
  Operation *getOperation() const;
  Location getLoc() const;
  StringRef getOperationName() const;
  EClassOp getEClassOp() const;
  ArrayRef<unsigned> getYieldedResultSlots() const;

  EValue getOperand(unsigned index) const;
  bool hasResult(unsigned index) const;
  EValue getResult(unsigned index) const;

private:
  friend bool operator==(EOpRefBase lhs, EOpRefBase rhs);
  friend class EGraph;
  friend class EGraphIndex;

  EGraph *graph = nullptr;
  EOpOccurrenceId id = 0;
  unsigned generation = 0;
};

inline bool operator==(EOpRefBase lhs, EOpRefBase rhs) {
  return lhs.graph == rhs.graph && lhs.id == rhs.id &&
         lhs.generation == rhs.generation;
}

inline bool operator!=(EOpRefBase lhs, EOpRefBase rhs) { return !(lhs == rhs); }

template <typename OpTy>
class EOpRef : public EOpRefBase {
public:
  EOpRef() = default;
  EOpRef(EOpRefBase base) : EOpRefBase(base) {
    assert((!base || llvm::isa<OpTy>(base.getOperation())) &&
           "typed EOpRef requires a matching payload operation");
  }

  OpTy getOp() const { return cast<OpTy>(getOperation()); }
};

/// Symbol-based handle for an e-class result slot.
class EValue {
public:
  EValue() = default;
  EValue(EGraph *graph, StringAttr symbolName, unsigned resultIndex = 0)
      : graph(graph), symbolName(symbolName), resultIndex(resultIndex) {}
  EValue(EGraph *graph, FlatSymbolRefAttr symbolRef, unsigned resultIndex = 0)
      : graph(graph),
        symbolName(symbolRef ? symbolRef.getAttr() : StringAttr()),
        resultIndex(resultIndex) {}

  EGraph *getGraph() const { return graph; }
  StringAttr getSymbolNameAttr() const { return symbolName; }
  StringRef getSymbolName() const {
    return symbolName ? symbolName.getValue() : StringRef();
  }
  FlatSymbolRefAttr getSymbolRef() const {
    return symbolName ? FlatSymbolRefAttr::get(symbolName)
                      : FlatSymbolRefAttr();
  }
  unsigned getResultIndex() const;
  Type getType() const;
  EValue getLeader() const;
  bool isEquivalentTo(EValue other) const;
  SmallVector<EOpRefBase> getDefs() const;

  template <typename OpTy>
  bool hasDef() const;

  template <typename OpTy>
  SmallVector<EOpRef<OpTy>> getDefs() const;

  template <typename OpTy, typename Fn>
  LogicalResult matchDef(Fn &&fn) const;

  template <typename OpTy>
  FailureOr<EOpRef<OpTy>> getUniqueDef() const;

  explicit operator bool() const { return graph && (symbolName || keyValue); }

private:
  friend class EGraph;
  friend class EOpRefBase;
  friend class EGraphRewriteTransaction;
  friend bool operator==(EValue lhs, EValue rhs);

  EValue(EGraph *graph, Value keyValue) : graph(graph), keyValue(keyValue) {}

  EGraph *graph = nullptr;
  StringAttr symbolName;
  unsigned resultIndex = 0;
  Value keyValue;
};

struct EGraphInternedOperation {
  Operation *operation = nullptr;
  SmallVector<Operation *, 4> sourceOperations;
  EClassOp eclass;
  SmallVector<EValue, 4> results;
  bool inserted = false;
};

struct EGraphUnionResult {
  bool changed = false;
  SmallVector<EValue, 4> leaderValues;
  SmallVector<EClassOp, 4> touchedEClasses;
};

inline bool operator==(EValue lhs, EValue rhs) {
  return lhs.graph == rhs.graph && lhs.symbolName == rhs.symbolName &&
         lhs.resultIndex == rhs.resultIndex && lhs.keyValue == rhs.keyValue;
}

inline bool operator!=(EValue lhs, EValue rhs) { return !(lhs == rhs); }

struct EGraphRebuildResult {
  SmallVector<EOpRefBase, 4> newCandidateRoots;
  SmallVector<EOpRefBase, 4> affectedParentCandidates;
};

/// Indexed view of the persistent egraph IR and live candidate occurrences.
class EGraphIndex {
public:
  static constexpr EOpOccurrenceId kInvalidOccurrenceId = ~EOpOccurrenceId(0);
  static constexpr unsigned kInvalidResultSlot = ~0u;

  struct OperationOccurrence {
    Operation *operation = nullptr;
    EClassOp eclass;
    unsigned candidateOrdinal = 0;
    SmallVector<unsigned> candidateArgInputIndices;
    SmallVector<StringAttr, 4> candidateArgSymbols;
    SmallVector<unsigned> resultSlotForOpResult;
    SmallVector<StringAttr, 4> resultSymbolsForOpResult;
    SmallVector<unsigned> yieldedResultSlots;
    unsigned generation = 0;
    bool live = false;
  };

  struct ValueLookup {
    OperationOwnership ownership = OperationOwnership::IllegalExternal;
    Value keyValue;
    bool legalAlias = false;
  };

  void clear();
  EOpOccurrenceId addOperationOccurrence(OperationOccurrence occurrence);
  void addEGraphValue(Value value, Value keyValue, bool legalAlias = false);
  void addScratchOperation(Operation *operation);
  void removeScratchOperation(Operation *operation);
  bool addStructuralEntry(EGraphStructuralKey key, EGraphHashConsEntry entry);
  void addCandidateRootForResult(Value resultKeyValue,
                                 EOpOccurrenceId occurrenceId);
  void addCandidateRootForSymbol(StringAttr resultSymbolName,
                                 EOpOccurrenceId occurrenceId);
  void addParentCandidate(Value childKeyValue, EOpOccurrenceId occurrenceId);
  void addParentCandidate(StringAttr childSymbolName,
                          EOpOccurrenceId occurrenceId);
  void markOccurrenceDead(EOpOccurrenceId id);

  const OperationOccurrence *lookup(EOpOccurrenceId id) const;
  OperationOccurrence *lookup(EOpOccurrenceId id);
  FailureOr<EOpOccurrenceId> lookupOperation(Operation *operation) const;
  const ValueLookup *lookupValue(Value value) const;
  const EGraphHashConsEntry *
  lookupStructuralEntry(const EGraphStructuralKey &key) const;
  ArrayRef<EOpOccurrenceId> getCandidateRootIds(EClassOp eclass) const;
  ArrayRef<EOpOccurrenceId> getCandidateRootIds(Value resultKeyValue) const;
  ArrayRef<EOpOccurrenceId>
  getCandidateRootIds(StringAttr resultSymbolName) const;
  ArrayRef<EOpOccurrenceId> getParentCandidateIds(Value childKeyValue) const;
  ArrayRef<EOpOccurrenceId>
  getParentCandidateIds(StringAttr childSymbolName) const;

  OperationOwnership getOperationOwnership(Operation *operation) const;
  OperationOwnership getValueOwnership(Value value) const;
  bool isLegalAliasValue(Value value) const;
  bool isLive(EOpRefBase ref) const;
  unsigned getGeneration(EOpOccurrenceId id) const;

  ArrayRef<OperationOccurrence> getOccurrences() const { return occurrences; }

private:
  struct OperationLookup {
    OperationOwnership ownership = OperationOwnership::IllegalExternal;
    EOpOccurrenceId occurrenceId = kInvalidOccurrenceId;
  };

  SmallVector<OperationOccurrence, 0> occurrences;
  DenseMap<Operation *, OperationLookup> operationLookup;
  DenseMap<Value, ValueLookup> valueLookup;
  DenseMap<EGraphStructuralKey, EGraphHashConsEntry> structuralHashCons;
  DenseMap<Operation *, SmallVector<EOpOccurrenceId>> candidateRootsByEClass;
  DenseMap<Value, SmallVector<EOpOccurrenceId>> candidateRootsByResultKey;
  DenseMap<StringAttr, SmallVector<EOpOccurrenceId>>
      candidateRootsByResultSymbol;
  DenseMap<Value, SmallVector<EOpOccurrenceId>> parentCandidatesByChild;
  DenseMap<StringAttr, SmallVector<EOpOccurrenceId>>
      parentCandidatesByChildSymbol;
  unsigned nextGeneration = 1;
};

/// High-level EGraph facade for lookup, intern, union, and rebuild.
class EGraph {
public:
  EGraph() = default;

  EValue getValue(StringAttr symbolName, unsigned resultIndex = 0) {
    return EValue(this, symbolName, resultIndex);
  }
  EValue getValue(FlatSymbolRefAttr symbolRef, unsigned resultIndex = 0) {
    return EValue(this, symbolRef, resultIndex);
  }

  void clearIndex() {
    index.clear();
    indexedEGraph = {};
    symbolicValueLookup.clear();
    payloadSymbols.clear();
  }
  LogicalResult indexEGraph(EGraphOp egraph);
  LogicalResult indexEClass(EClassOp eclass);
  void registerBoundaryValue(Value value);
  void registerScratchOperation(Operation *operation);
  void unregisterScratchOperation(Operation *operation);

  SmallVector<EOpRefBase> getOpRefs();
  SmallVector<EOpRefBase> getCandidateRoots(EClassOp eclass);
  SmallVector<EOpRefBase> getCandidateRoots(EValue value);
  SmallVector<EOpRefBase> getParentCandidates(EValue child);

  OperationOwnership getOperationOwnership(Operation *operation) const {
    return index.getOperationOwnership(operation);
  }

  OperationOwnership getValueOwnership(Value value) const {
    if (symbolicValueLookup.contains(value))
      return OperationOwnership::EGraphOwned;
    return index.getValueOwnership(value);
  }

  bool isLegalAliasValue(Value value) const {
    if (index.isLegalAliasValue(value))
      return true;

    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument)
      return false;

    Operation *parentOp = argument.getOwner()->getParentOp();
    return parentOp && isa<EClassOp>(parentOp);
  }

  FailureOr<EOpRefBase> lookupOpRef(Operation *operation);
  FailureOr<EValue> lookupValue(Value value) const;
  FailureOr<unsigned> getInputIndex(EClassOp eclass, EValue value);
  FailureOr<unsigned> getResultIndex(EClassOp eclass, EValue value);
  FailureOr<EGraphStructuralKey> getStructuralKey(EOpRefBase ref);
  const EGraphHashConsEntry *
  lookupStructuralEntry(const EGraphStructuralKey &key) const {
    return index.lookupStructuralEntry(key);
  }
  FailureOr<EGraphHashConsEntry> lookupStructuralEntry(EOpRefBase ref);
  FailureOr<EGraphInternedOperation> intern(Operation *operation,
                                            ArrayRef<EValue> operands,
                                            Operation *insertionAnchor);
  FailureOr<EGraphUnionResult> unionValues(EValue lhs, EValue rhs);
  FailureOr<EGraphUnionResult> unionValueTuples(ArrayRef<EValue> lhs,
                                                ArrayRef<EValue> rhs);
  FailureOr<EGraphRebuildResult> rebuild(Operation *root);
  bool isClean() const {
    return touchedEClasses.empty() && touchedSymbolNames.empty();
  }
  ArrayRef<EClassOp> getTouchedEClasses() const { return touchedEClasses; }
  void clearTouchedEClasses() {
    touchedEClasses.clear();
    touchedSymbolNames.clear();
  }

  const EGraphIndex &getIndex() const { return index; }
  EGraphIndex &getIndex() { return index; }

  const EGraphIndex::OperationOccurrence &getOccurrence(EOpRefBase ref) const;
  EValue
  resolveValueInOccurrence(const EGraphIndex::OperationOccurrence &occurrence,
                           Value value);

private:
  friend class EValue;
  friend class EGraphRewriteTransaction;

  Operation *lookupPayloadSymbol(StringAttr symbolName) const;
  LogicalResult indexSymbolicEClass(EClassOp eclass);
  EValue getValueForKey(Value keyValue) const {
    return EValue(const_cast<EGraph *>(this), keyValue);
  }
  Value getLeaderKey(EValue value) const {
    return resolveLeaderValue(value).keyValue;
  }
  EValue resolveLeaderValue(EValue value) const;
  StringAttr findLeaderSymbol(StringAttr symbolName) const;
  FailureOr<StringAttr> getLeaderSymbolForHashing(EValue value) const;
  FailureOr<EValue> tryResolveValueInOccurrence(
      const EGraphIndex::OperationOccurrence &occurrence, Value value);
  FailureOr<EGraphStructuralKey>
  buildStructuralKey(const EGraphIndex::OperationOccurrence &occurrence);
  EGraphHashConsEntry
  buildHashConsEntry(const EGraphIndex::OperationOccurrence &occurrence);
  LogicalResult markDuplicateOccurrencesDead();
  bool isSymbolicAliasCandidate(EClassOp eclass,
                                unsigned candidateOrdinal) const;
  FailureOr<EClassOp>
  recreateSymbolicEClass(EClassOp eclass, ArrayAttr candidateRefs,
                         ArrayRef<EClassOp> absorbedMembers = {},
                         bool preserveExistingCandidates = true,
                         unsigned extraCandidateCount = 0);
  FailureOr<bool> hasSymbolicAliasCandidate(EClassOp eclass,
                                            StringAttr memberSymbol) const;
  LogicalResult rewriteSymbolicCandidateRefsToLeaders(EGraphOp egraph);
  LogicalResult appendSymbolicAliasCandidate(EClassOp leader,
                                             StringAttr memberSymbol);
  LogicalResult materializeSymbolicAlias(EClassOp leader,
                                         StringAttr memberSymbol);
  LogicalResult materializeSymbolicUnions(EGraphOp egraph);
  LogicalResult rewriteReturnTargetsToLeaders(EGraphOp egraph);
  LogicalResult removeDeadSymbolicEClasses(EGraphOp egraph);
  FailureOr<EGraphUnionResult> unionSymbolicStructuralCollisions();
  Value findLeaderKey(Value keyValue) const;
  FailureOr<std::pair<StringAttr, StringAttr>>
  getUnionSymbolPair(EValue lhs, EValue rhs) const;
  FailureOr<std::pair<Value, Value>> getUnionKeyPair(EValue lhs,
                                                     EValue rhs) const;
  EGraphUnionResult
  unionLeaderSymbols(ArrayRef<std::pair<StringAttr, StringAttr>> symbolPairs);
  EGraphUnionResult
  unionLeaderKeyPairs(ArrayRef<std::pair<Value, Value>> keyPairs);
  void recordTouchedSymbol(StringAttr symbolName, EGraphUnionResult &result);
  void recordTouchedEClass(Value keyValue, EGraphUnionResult &result);

  EGraphIndex index;
  EGraphOp indexedEGraph;
  DenseMap<Value, StringAttr> symbolicValueLookup;
  DenseMap<StringAttr, Operation *> payloadSymbols;
  DenseMap<StringAttr, StringAttr> symbolicUnionParents;
  DenseMap<Value, Value> unionParents;
  SmallVector<EClassOp, 4> touchedEClasses;
  SmallVector<StringAttr, 4> touchedSymbolNames;
};

template <typename OpTy>
bool EValue::hasDef() const {
  return !getDefs<OpTy>().empty();
}

template <typename OpTy>
SmallVector<EOpRef<OpTy>> EValue::getDefs() const {
  SmallVector<EOpRef<OpTy>> refs;
  SmallVector<EOpRefBase> allDefs = EValue::getDefs();
  for (EOpRefBase ref : allDefs) {
    if (!llvm::isa<OpTy>(ref.getOperation()))
      continue;
    refs.push_back(EOpRef<OpTy>(ref));
  }
  return refs;
}

template <typename OpTy, typename Fn>
LogicalResult EValue::matchDef(Fn &&fn) const {
  using CallbackResult = std::invoke_result_t<Fn &, EOpRef<OpTy>>;
  static_assert(std::is_same_v<std::decay_t<CallbackResult>, LogicalResult> ||
                    std::is_same_v<std::decay_t<CallbackResult>, bool> ||
                    std::is_void_v<CallbackResult>,
                "matchDef callback must return LogicalResult, bool, or void");

  for (EOpRef<OpTy> ref : getDefs<OpTy>()) {
    if constexpr (std::is_same_v<std::decay_t<CallbackResult>, LogicalResult>) {
      if (succeeded(fn(ref)))
        return success();
    } else if constexpr (std::is_same_v<std::decay_t<CallbackResult>, bool>) {
      if (fn(ref))
        return success();
    } else {
      fn(ref);
    }
  }
  return failure();
}

template <typename OpTy>
FailureOr<EOpRef<OpTy>> EValue::getUniqueDef() const {
  SmallVector<EOpRef<OpTy>> defs = getDefs<OpTy>();
  if (defs.size() != 1)
    return failure();
  return defs.front();
}

} // namespace egraph
} // namespace mlir

#endif // MLIR_EGRAPH_EGRAPH_EGRAPH_H
