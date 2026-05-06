#include "MLIREGraph/EGraph/EGraph.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <cctype>

using namespace mlir;
using namespace mlir::egraph;

namespace {
Type getPayloadTypeForKey(Value keyValue) { return keyValue.getType(); }

Type getPayloadTypeForSymbol(Operation *symbolOp) {
  if (auto input = dyn_cast_or_null<InputOp>(symbolOp))
    return input.getPayloadType();
  if (auto eclass = dyn_cast_or_null<EClassOp>(symbolOp))
    return eclass.getPayloadType();
  return Type();
}

std::string sanitizeSymbolStem(StringRef stem) {
  std::string sanitized;
  sanitized.reserve(stem.size());
  for (char ch : stem) {
    unsigned char byte = static_cast<unsigned char>(ch);
    if (std::isalnum(byte) || ch == '_') {
      sanitized.push_back(ch);
      continue;
    }
    if (sanitized.empty() || sanitized.back() == '_')
      continue;
    sanitized.push_back('_');
  }

  while (!sanitized.empty() && sanitized.back() == '_')
    sanitized.pop_back();

  if (sanitized.empty())
    return "value";
  if (std::isdigit(static_cast<unsigned char>(sanitized.front())))
    sanitized.insert(sanitized.begin(), 's');
  return sanitized;
}

std::string getOperationStem(OperationName operationName) {
  StringRef name = operationName.getStringRef();
  size_t dot = name.rfind('.');
  if (dot != StringRef::npos)
    name = name.drop_front(dot + 1);
  return sanitizeSymbolStem(name);
}

LogicalResult
readCandidateSymbolRefs(EClassOp eclass, unsigned candidateOrdinal,
                        SmallVectorImpl<FlatSymbolRefAttr> &refs) {
  ArrayAttr candidateRefs = eclass.getCandidateRefs();
  if (candidateOrdinal >= candidateRefs.size())
    return eclass.emitOpError("missing candidate_refs entry #")
           << candidateOrdinal;

  auto row = dyn_cast<ArrayAttr>(candidateRefs[candidateOrdinal]);
  if (!row)
    return eclass.emitOpError("candidate_refs entry #")
           << candidateOrdinal << " must be a symbol ref array attribute";

  refs.clear();
  refs.reserve(row.size());
  for (Attribute attr : row) {
    auto symbolRef = dyn_cast<FlatSymbolRefAttr>(attr);
    if (!symbolRef)
      return eclass.emitOpError("candidate_refs entry #")
             << candidateOrdinal << " must be a symbol ref array attribute";
    refs.push_back(symbolRef);
  }

  return success();
}

SmallVector<unsigned> getYieldedResultSlots(ArrayRef<unsigned> resultSlots) {
  SmallVector<unsigned> yieldedSlots;
  for (unsigned slot : resultSlots)
    if (slot != EGraphIndex::kInvalidResultSlot)
      yieldedSlots.push_back(slot);
  return yieldedSlots;
}

FailureOr<unsigned> getEClassResultSlot(Value keyValue) {
  auto result = dyn_cast<OpResult>(keyValue);
  if (!result)
    return failure();

  if (!isa<EClassOp>(result.getOwner()))
    return failure();

  return result.getResultNumber();
}

LogicalResult validateUnionKeyPair(Value lhsKeyValue, Value rhsKeyValue) {
  if (!lhsKeyValue || !rhsKeyValue ||
      getPayloadTypeForKey(lhsKeyValue) != getPayloadTypeForKey(rhsKeyValue))
    return failure();

  FailureOr<unsigned> lhsSlot = getEClassResultSlot(lhsKeyValue);
  FailureOr<unsigned> rhsSlot = getEClassResultSlot(rhsKeyValue);
  if (succeeded(lhsSlot) && succeeded(rhsSlot) && *lhsSlot != *rhsSlot)
    return failure();

  return success();
}
} // namespace

Type EValue::getType() const {
  if (symbolName) {
    assert(graph && "cannot query a symbol-backed EValue without an EGraph");
    if (resultIndex != 0)
      return Type();
    Operation *symbolOp = graph->lookupPayloadSymbol(symbolName);
    return getPayloadTypeForSymbol(symbolOp);
  }
  if (!keyValue)
    return Type();
  return getPayloadTypeForKey(keyValue);
}

unsigned EValue::getResultIndex() const {
  if (symbolName)
    return resultIndex;

  FailureOr<unsigned> resultSlot = getEClassResultSlot(keyValue);
  if (succeeded(resultSlot))
    return *resultSlot;

  return 0;
}

EValue EValue::getLeader() const {
  assert(graph && "cannot resolve an EValue leader without an EGraph");
  return graph->resolveLeaderValue(*this);
}

bool EValue::isEquivalentTo(EValue other) const {
  if (!*this || !other || graph != other.graph)
    return false;
  return getLeader() == other.getLeader();
}

SmallVector<EOpRefBase> EValue::getDefs() const {
  if (!*this)
    return {};
  return graph->getCandidateRoots(*this);
}

bool EOpRefBase::isLive() const {
  return graph && graph->getIndex().isLive(*this);
}

bool EOpRefBase::isSameCandidateAs(EOpRefBase other) const {
  if (!graph || graph != other.graph || !isLive() || !other.isLive())
    return false;
  return getOperation()->getBlock()->getParent() ==
         other.getOperation()->getBlock()->getParent();
}

Operation *EOpRefBase::getOperation() const {
  return graph->getOccurrence(*this).operation;
}

Location EOpRefBase::getLoc() const { return getOperation()->getLoc(); }

StringRef EOpRefBase::getOperationName() const {
  return getOperation()->getName().getStringRef();
}

EClassOp EOpRefBase::getEClassOp() const {
  return graph->getOccurrence(*this).eclass;
}

ArrayRef<unsigned> EOpRefBase::getYieldedResultSlots() const {
  return graph->getOccurrence(*this).yieldedResultSlots;
}

EValue EOpRefBase::getOperand(unsigned index) const {
  const EGraphIndex::OperationOccurrence &occurrence =
      graph->getOccurrence(*this);
  assert(index < occurrence.operation->getNumOperands() &&
         "operand index is out of range");
  return graph->resolveValueInOccurrence(
      occurrence, occurrence.operation->getOperand(index));
}

bool EOpRefBase::hasResult(unsigned index) const {
  const EGraphIndex::OperationOccurrence &occurrence =
      graph->getOccurrence(*this);
  if (index >= occurrence.resultSlotForOpResult.size())
    return false;
  if (occurrence.resultSlotForOpResult[index] !=
      EGraphIndex::kInvalidResultSlot)
    return true;
  return index < occurrence.resultSymbolsForOpResult.size() &&
         occurrence.resultSymbolsForOpResult[index];
}

EValue EOpRefBase::getResult(unsigned index) const {
  const EGraphIndex::OperationOccurrence &occurrence =
      graph->getOccurrence(*this);
  assert(index < occurrence.resultSlotForOpResult.size() &&
         "result index is out of range");
  if (index < occurrence.resultSymbolsForOpResult.size()) {
    StringAttr symbolName = occurrence.resultSymbolsForOpResult[index];
    if (symbolName)
      return EValue(graph, symbolName);
  }

  unsigned resultSlot = occurrence.resultSlotForOpResult[index];
  assert(resultSlot != EGraphIndex::kInvalidResultSlot &&
         "operation result is not yielded by this occurrence");
  auto symbolName = occurrence.eclass->getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  return EValue(graph, symbolName, resultSlot);
}

void EGraphIndex::clear() {
  occurrences.clear();
  operationLookup.clear();
  valueLookup.clear();
  structuralHashCons.clear();
  candidateRootsByEClass.clear();
  candidateRootsByResultKey.clear();
  candidateRootsByResultSymbol.clear();
  parentCandidatesByChild.clear();
  parentCandidatesByChildSymbol.clear();
  // Keep generations monotonic so refs from before a rebuild stay stale.
}

EOpOccurrenceId
EGraphIndex::addOperationOccurrence(OperationOccurrence occurrence) {
  EOpOccurrenceId id = occurrences.size();
  occurrence.generation = nextGeneration++;
  occurrence.live = true;
  Operation *operation = occurrence.operation;
  Operation *eclassOperation = occurrence.eclass.getOperation();
  occurrences.push_back(std::move(occurrence));
  operationLookup[operation] = {OperationOwnership::EGraphOwned, id};
  candidateRootsByEClass[eclassOperation].push_back(id);
  return id;
}

void EGraphIndex::addEGraphValue(Value value, Value keyValue, bool legalAlias) {
  valueLookup[value] = {OperationOwnership::EGraphOwned, keyValue, legalAlias};
}

void EGraphIndex::addScratchOperation(Operation *operation) {
  operationLookup[operation] = {OperationOwnership::ScratchCreated,
                                kInvalidOccurrenceId};
  for (Value result : operation->getResults())
    valueLookup[result] = {OperationOwnership::ScratchCreated, Value(),
                           /*legalAlias=*/false};
}

void EGraphIndex::removeScratchOperation(Operation *operation) {
  auto operationIt = operationLookup.find(operation);
  if (operationIt != operationLookup.end() &&
      operationIt->second.ownership == OperationOwnership::ScratchCreated)
    operationLookup.erase(operationIt);

  for (Value result : operation->getResults()) {
    auto valueIt = valueLookup.find(result);
    if (valueIt != valueLookup.end() &&
        valueIt->second.ownership == OperationOwnership::ScratchCreated)
      valueLookup.erase(valueIt);
  }
}

bool EGraphIndex::addStructuralEntry(EGraphStructuralKey key,
                                     EGraphHashConsEntry entry) {
  return structuralHashCons.insert({std::move(key), std::move(entry)}).second;
}

void EGraphIndex::addCandidateRootForResult(Value resultKeyValue,
                                            EOpOccurrenceId occurrenceId) {
  candidateRootsByResultKey[resultKeyValue].push_back(occurrenceId);
}

void EGraphIndex::addCandidateRootForSymbol(StringAttr resultSymbolName,
                                            EOpOccurrenceId occurrenceId) {
  candidateRootsByResultSymbol[resultSymbolName].push_back(occurrenceId);
}

void EGraphIndex::addParentCandidate(Value childKeyValue,
                                     EOpOccurrenceId occurrenceId) {
  parentCandidatesByChild[childKeyValue].push_back(occurrenceId);
}

void EGraphIndex::addParentCandidate(StringAttr childSymbolName,
                                     EOpOccurrenceId occurrenceId) {
  parentCandidatesByChildSymbol[childSymbolName].push_back(occurrenceId);
}

void EGraphIndex::markOccurrenceDead(EOpOccurrenceId id) {
  OperationOccurrence *occurrence = lookup(id);
  if (occurrence)
    occurrence->live = false;
}

const EGraphIndex::OperationOccurrence *
EGraphIndex::lookup(EOpOccurrenceId id) const {
  if (id >= occurrences.size())
    return nullptr;
  return &occurrences[id];
}

EGraphIndex::OperationOccurrence *EGraphIndex::lookup(EOpOccurrenceId id) {
  if (id >= occurrences.size())
    return nullptr;
  return &occurrences[id];
}

FailureOr<EOpOccurrenceId>
EGraphIndex::lookupOperation(Operation *operation) const {
  auto it = operationLookup.find(operation);
  if (it == operationLookup.end() ||
      it->second.ownership != OperationOwnership::EGraphOwned)
    return failure();

  const OperationOccurrence *occurrence = lookup(it->second.occurrenceId);
  if (!occurrence || !occurrence->live)
    return failure();

  return it->second.occurrenceId;
}

const EGraphIndex::ValueLookup *EGraphIndex::lookupValue(Value value) const {
  auto it = valueLookup.find(value);
  if (it == valueLookup.end())
    return nullptr;
  return &it->second;
}

const EGraphHashConsEntry *
EGraphIndex::lookupStructuralEntry(const EGraphStructuralKey &key) const {
  auto it = structuralHashCons.find(key);
  if (it == structuralHashCons.end())
    return nullptr;
  return &it->second;
}

ArrayRef<EOpOccurrenceId>
EGraphIndex::getCandidateRootIds(EClassOp eclass) const {
  auto it = candidateRootsByEClass.find(eclass.getOperation());
  if (it == candidateRootsByEClass.end())
    return {};
  return it->second;
}

ArrayRef<EOpOccurrenceId>
EGraphIndex::getCandidateRootIds(Value resultKeyValue) const {
  auto it = candidateRootsByResultKey.find(resultKeyValue);
  if (it == candidateRootsByResultKey.end())
    return {};
  return it->second;
}

ArrayRef<EOpOccurrenceId>
EGraphIndex::getCandidateRootIds(StringAttr resultSymbolName) const {
  auto it = candidateRootsByResultSymbol.find(resultSymbolName);
  if (it == candidateRootsByResultSymbol.end())
    return {};
  return it->second;
}

ArrayRef<EOpOccurrenceId>
EGraphIndex::getParentCandidateIds(Value childKeyValue) const {
  auto it = parentCandidatesByChild.find(childKeyValue);
  if (it == parentCandidatesByChild.end())
    return {};
  return it->second;
}

ArrayRef<EOpOccurrenceId>
EGraphIndex::getParentCandidateIds(StringAttr childSymbolName) const {
  auto it = parentCandidatesByChildSymbol.find(childSymbolName);
  if (it == parentCandidatesByChildSymbol.end())
    return {};
  return it->second;
}

OperationOwnership
EGraphIndex::getOperationOwnership(Operation *operation) const {
  auto it = operationLookup.find(operation);
  if (it == operationLookup.end())
    return OperationOwnership::IllegalExternal;
  return it->second.ownership;
}

OperationOwnership EGraphIndex::getValueOwnership(Value value) const {
  auto it = valueLookup.find(value);
  if (it == valueLookup.end())
    return OperationOwnership::IllegalExternal;
  return it->second.ownership;
}

bool EGraphIndex::isLegalAliasValue(Value value) const {
  auto it = valueLookup.find(value);
  return it != valueLookup.end() && it->second.legalAlias;
}

bool EGraphIndex::isLive(EOpRefBase ref) const {
  const OperationOccurrence *occurrence = lookup(ref.id);
  return occurrence && occurrence->live &&
         occurrence->generation == ref.generation;
}

unsigned EGraphIndex::getGeneration(EOpOccurrenceId id) const {
  const OperationOccurrence *occurrence = lookup(id);
  assert(occurrence && "invalid operation occurrence id");
  return occurrence->generation;
}

EValue EGraph::resolveLeaderValue(EValue value) const {
  assert(value.getGraph() == this &&
         "cannot resolve an EValue leader from another EGraph");
  if (value.getSymbolNameAttr())
    return EValue(value.getGraph(), findLeaderSymbol(value.getSymbolNameAttr()),
                  value.getResultIndex());
  return value;
}

FailureOr<StringAttr> EGraph::getLeaderSymbolForHashing(EValue value) const {
  if (value.getGraph() != this)
    return failure();

  EValue leader = resolveLeaderValue(value);
  if (leader.getSymbolNameAttr()) {
    if (leader.getResultIndex() != 0)
      return failure();
    return leader.getSymbolNameAttr();
  }
  return failure();
}

Operation *EGraph::lookupPayloadSymbol(StringAttr symbolName) const {
  if (!symbolName)
    return nullptr;

  auto cached = payloadSymbols.find(symbolName);
  if (cached != payloadSymbols.end())
    return cached->second;

  if (!indexedEGraph)
    return nullptr;

  return SymbolTable::lookupSymbolIn(indexedEGraph,
                                     FlatSymbolRefAttr::get(symbolName));
}

LogicalResult EGraph::indexEGraph(EGraphOp egraph) {
  if (!egraph || egraph.isExternal())
    return failure();

  clearIndex();
  indexedEGraph = egraph;

  Block &entryBlock = egraph.getBody().front();
  for (Operation &op : entryBlock) {
    if (auto input = dyn_cast<InputOp>(op)) {
      StringAttr symbolName = input.getSymNameAttr();
      payloadSymbols[symbolName] = input.getOperation();
      symbolicValueLookup[input.getValue()] = symbolName;
      continue;
    }

    auto eclass = dyn_cast<EClassOp>(op);
    if (!eclass)
      continue;
    if (failed(indexSymbolicEClass(eclass)))
      return failure();
  }

  return success();
}

LogicalResult EGraph::indexSymbolicEClass(EClassOp eclass) {
  StringAttr eclassName = eclass.getSymNameAttr();
  payloadSymbols[eclassName] = eclass.getOperation();

  for (auto indexedRegion : llvm::enumerate(eclass.getCandidates())) {
    unsigned candidateOrdinal = indexedRegion.index();
    Region &candidate = indexedRegion.value();
    if (candidate.empty())
      return eclass.emitOpError("candidate region #")
             << candidateOrdinal << " must contain a block";

    SmallVector<FlatSymbolRefAttr> refs;
    if (failed(readCandidateSymbolRefs(eclass, candidateOrdinal, refs)))
      return failure();

    Block &block = candidate.front();
    if (block.getNumArguments() != refs.size())
      return eclass.emitOpError("candidate region #")
             << candidateOrdinal << " has " << block.getNumArguments()
             << " block arguments but candidate_refs entry has " << refs.size()
             << " symbol refs";

    for (auto [arg, ref] : llvm::zip_equal(block.getArguments(), refs))
      symbolicValueLookup[arg] = ref.getAttr();

    auto yield = dyn_cast_or_null<YieldOp>(block.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return eclass.emitOpError("candidate region #")
             << candidateOrdinal
             << " must terminate with a single-result egraph.yield";

    auto yieldedResult = dyn_cast<OpResult>(yield.getOperand(0));
    DenseMap<Operation *, SmallVector<unsigned>> resultSlotsByOperation;
    if (yieldedResult && yieldedResult.getOwner()->getBlock() == &block) {
      symbolicValueLookup[yieldedResult] = eclassName;

      SmallVector<unsigned> &resultSlots =
          resultSlotsByOperation[yieldedResult.getOwner()];
      resultSlots.resize(yieldedResult.getOwner()->getNumResults(),
                         EGraphIndex::kInvalidResultSlot);
      resultSlots[yieldedResult.getResultNumber()] = 0;
    }

    for (Operation &operation : block) {
      if (isa<YieldOp>(operation))
        continue;

      auto it = resultSlotsByOperation.find(&operation);
      if (it == resultSlotsByOperation.end())
        continue;

      EGraphIndex::OperationOccurrence occurrence;
      occurrence.operation = &operation;
      occurrence.eclass = eclass;
      occurrence.candidateOrdinal = candidateOrdinal;
      occurrence.resultSlotForOpResult = it->second;
      occurrence.resultSymbolsForOpResult.resize(operation.getNumResults());
      for (FlatSymbolRefAttr ref : refs)
        occurrence.candidateArgSymbols.push_back(ref.getAttr());
      for (auto indexedResult :
           llvm::enumerate(occurrence.resultSlotForOpResult)) {
        if (indexedResult.value() == EGraphIndex::kInvalidResultSlot)
          continue;
        occurrence.resultSymbolsForOpResult[indexedResult.index()] = eclassName;
      }
      occurrence.yieldedResultSlots =
          getYieldedResultSlots(occurrence.resultSlotForOpResult);
      EOpOccurrenceId occurrenceId =
          index.addOperationOccurrence(std::move(occurrence));
      const EGraphIndex::OperationOccurrence *storedOccurrence =
          index.lookup(occurrenceId);
      FailureOr<EGraphStructuralKey> structuralKey =
          buildStructuralKey(*storedOccurrence);
      if (failed(structuralKey))
        return operation.emitOpError("could not build egraph structural key");
      index.addStructuralEntry(std::move(*structuralKey),
                               buildHashConsEntry(*storedOccurrence));
      index.addCandidateRootForSymbol(findLeaderSymbol(eclassName),
                                      occurrenceId);

      SmallVector<StringAttr, 4> indexedChildSymbols;
      for (FlatSymbolRefAttr ref : refs) {
        StringAttr childSymbol = findLeaderSymbol(ref.getAttr());
        if (llvm::is_contained(indexedChildSymbols, childSymbol))
          continue;
        indexedChildSymbols.push_back(childSymbol);
        index.addParentCandidate(childSymbol, occurrenceId);
      }
    }
  }

  return success();
}

void EGraph::registerScratchOperation(Operation *operation) {
  index.addScratchOperation(operation);
}

void EGraph::unregisterScratchOperation(Operation *operation) {
  index.removeScratchOperation(operation);
}

SmallVector<EOpRefBase> EGraph::getOpRefs() {
  SmallVector<EOpRefBase> refs;
  for (auto indexedOccurrence : llvm::enumerate(index.getOccurrences())) {
    const EGraphIndex::OperationOccurrence &occurrence =
        indexedOccurrence.value();
    if (!occurrence.live)
      continue;
    refs.emplace_back(this, indexedOccurrence.index(), occurrence.generation);
  }
  return refs;
}

SmallVector<EOpRefBase> EGraph::getCandidateRoots(EClassOp eclass) {
  SmallVector<EOpRefBase> refs;
  for (EOpOccurrenceId id : index.getCandidateRootIds(eclass)) {
    const EGraphIndex::OperationOccurrence *occurrence = index.lookup(id);
    if (!occurrence || !occurrence->live)
      continue;
    refs.emplace_back(this, id, occurrence->generation);
  }
  return refs;
}

SmallVector<EOpRefBase> EGraph::getCandidateRoots(EValue value) {
  SmallVector<EOpRefBase> refs;
  if (value.getGraph() != this || !value.getSymbolNameAttr())
    return refs;

  EValue leader = value.getLeader();
  if (leader.getResultIndex() != 0)
    return refs;
  for (EOpOccurrenceId id :
       index.getCandidateRootIds(leader.getSymbolNameAttr())) {
    const EGraphIndex::OperationOccurrence *occurrence = index.lookup(id);
    if (!occurrence || !occurrence->live)
      continue;
    refs.emplace_back(this, id, occurrence->generation);
  }
  return refs;
}

SmallVector<EOpRefBase> EGraph::getParentCandidates(EValue child) {
  SmallVector<EOpRefBase> refs;
  if (child.getGraph() != this || !child.getSymbolNameAttr())
    return refs;

  EValue leader = child.getLeader();
  if (leader.getResultIndex() != 0)
    return refs;
  for (EOpOccurrenceId id :
       index.getParentCandidateIds(leader.getSymbolNameAttr())) {
    const EGraphIndex::OperationOccurrence *occurrence = index.lookup(id);
    if (!occurrence || !occurrence->live)
      continue;
    refs.emplace_back(this, id, occurrence->generation);
  }
  return refs;
}

const EGraphIndex::OperationOccurrence &
EGraph::getOccurrence(EOpRefBase ref) const {
  assert(ref.getGraph() == this &&
         "operation occurrence belongs to another EGraph");
  const EGraphIndex::OperationOccurrence *occurrence = index.lookup(ref.id);
  assert(occurrence && "invalid operation occurrence id");
  assert(occurrence->generation == ref.generation &&
         "stale operation occurrence reference");
  return *occurrence;
}

FailureOr<EOpRefBase> EGraph::lookupOpRef(Operation *operation) {
  FailureOr<EOpOccurrenceId> id = index.lookupOperation(operation);
  if (failed(id))
    return failure();

  return EOpRefBase(this, *id, index.getGeneration(*id));
}

FailureOr<EValue> EGraph::lookupValue(Value value) const {
  auto symbolicLookup = symbolicValueLookup.find(value);
  if (symbolicLookup != symbolicValueLookup.end())
    return resolveLeaderValue(
        EValue(const_cast<EGraph *>(this), symbolicLookup->second));
  return failure();
}

FailureOr<unsigned> EGraph::getInputIndex(EClassOp eclass, EValue value) {
  if (!eclass || value.getGraph() != this || !value.getSymbolNameAttr() ||
      value.getResultIndex() != 0)
    return failure();

  StringAttr targetLeader = value.getLeader().getSymbolNameAttr();
  SmallVector<StringAttr, 4> seenSymbols;
  // candidate_refs are stored positionally, so recovering the external input
  // index requires scanning the flattened child symbol sequence in order.
  for (auto indexedRegion : llvm::enumerate(eclass.getCandidates())) {
    SmallVector<FlatSymbolRefAttr> refs;
    if (failed(readCandidateSymbolRefs(eclass, indexedRegion.index(), refs)))
      return failure();

    for (FlatSymbolRefAttr ref : refs) {
      StringAttr leader = findLeaderSymbol(ref.getAttr());
      if (llvm::is_contained(seenSymbols, leader))
        continue;
      if (leader == targetLeader)
        return seenSymbols.size();
      seenSymbols.push_back(leader);
    }
  }

  return failure();
}

FailureOr<unsigned> EGraph::getResultIndex(EClassOp eclass, EValue value) {
  if (!eclass || value.getGraph() != this || !value.getSymbolNameAttr())
    return failure();

  StringAttr eclassLeader = findLeaderSymbol(eclass.getSymNameAttr());
  // Symbolic eclasses currently expose only slot 0; other slots are reserved
  // for future multi-result support and must not participate in lookup.
  EValue leader = value.getLeader();
  if (leader.getSymbolNameAttr() == eclassLeader &&
      leader.getResultIndex() == 0)
    return FailureOr<unsigned>(0u);
  return failure();
}

FailureOr<EGraphStructuralKey> EGraph::getStructuralKey(EOpRefBase ref) {
  if (ref.getGraph() != this || !index.isLive(ref))
    return failure();

  const EGraphIndex::OperationOccurrence *occurrence = index.lookup(ref.id);
  if (!occurrence)
    return failure();
  return buildStructuralKey(*occurrence);
}

FailureOr<EGraphHashConsEntry> EGraph::lookupStructuralEntry(EOpRefBase ref) {
  FailureOr<EGraphStructuralKey> key = getStructuralKey(ref);
  if (failed(key))
    return failure();

  const EGraphHashConsEntry *entry = index.lookupStructuralEntry(*key);
  if (!entry)
    return failure();
  return *entry;
}

FailureOr<EGraphInternedOperation> EGraph::intern(Operation *operation,
                                                  ArrayRef<EValue> operands,
                                                  Operation *insertionAnchor) {
  if (!operation || operation->getNumOperands() != operands.size() ||
      !insertionAnchor || operation->getNumRegions() != 0 ||
      operation->getNumSuccessors() != 0)
    return failure();

  EGraphStructuralKey key;
  key.operationName = operation->getName();
  key.attributes = operation->getAttrDictionary();
  for (Type resultType : operation->getResultTypes())
    key.resultTypes.push_back(resultType);

  EGraphInternedOperation result;
  result.operation = operation;
  result.sourceOperations.push_back(operation);

  if (!indexedEGraph || operation->getNumResults() != 1)
    return failure();

  SmallVector<FlatSymbolRefAttr, 4> childRefs;
  childRefs.reserve(operands.size());
  for (EValue operand : operands) {
    if (!operand || operand.getGraph() != this)
      return failure();

    FailureOr<StringAttr> leaderSymbol = getLeaderSymbolForHashing(operand);
    if (failed(leaderSymbol))
      return failure();
    key.childLeaderSymbols.push_back(*leaderSymbol);
    childRefs.push_back(FlatSymbolRefAttr::get(*leaderSymbol));
  }

  if (const EGraphHashConsEntry *entry = lookupStructuralEntry(key)) {
    result.eclass = entry->eclass;
    result.inserted = false;
    if (entry->resultSymbols.empty() || !entry->resultSymbols.front())
      return failure();
    result.results.push_back(getValue(entry->resultSymbols.front()));
    return result;
  }

  auto makeFreshSymbolName = [&]() -> StringAttr {
    Builder builder(indexedEGraph.getContext());
    std::string base = "__intern_" + getOperationStem(operation->getName());
    std::string candidate = base;
    unsigned suffix = 0;
    while (true) {
      StringAttr candidateAttr = builder.getStringAttr(candidate);
      if (!payloadSymbols.contains(candidateAttr) &&
          !symbolicUnionParents.contains(candidateAttr) &&
          !SymbolTable::lookupSymbolIn(indexedEGraph,
                                       FlatSymbolRefAttr::get(candidateAttr)))
        return candidateAttr;
      candidate = base + "_" + std::to_string(++suffix);
    }
  };

  OpBuilder builder(insertionAnchor->getContext());
  builder.setInsertionPointAfter(insertionAnchor);

  SmallVector<Attribute, 4> candidateRefAttrs(childRefs.begin(),
                                              childRefs.end());
  auto eclass = EClassOp::create(
      builder, operation->getLoc(), makeFreshSymbolName(),
      TypeAttr::get(operation->getResult(0).getType()),
      builder.getArrayAttr({builder.getArrayAttr(candidateRefAttrs)}),
      /*candidatesCount=*/1);

  Region &candidate = eclass.getCandidates().front();
  Block &block = candidate.emplaceBlock();
  SmallVector<Value, 4> clonedOperands;
  clonedOperands.reserve(operands.size());
  for (EValue operand : operands)
    clonedOperands.push_back(
        block.addArgument(operand.getType(), operation->getLoc()));

  OpBuilder bodyBuilder(&block, block.begin());
  IRMapping mapper;
  for (auto [oldOperand, newOperand] :
       llvm::zip_equal(operation->getOperands(), clonedOperands))
    mapper.map(oldOperand, newOperand);
  Operation *candidateOperation = bodyBuilder.insert(operation->clone(mapper));
  YieldOp::create(bodyBuilder, operation->getLoc(),
                  candidateOperation->getResult(0));

  if (failed(indexSymbolicEClass(eclass))) {
    eclass.erase();
    return failure();
  }

  result.eclass = eclass;
  result.inserted = true;
  result.results.push_back(getValue(eclass.getSymNameAttr()));
  return result;
}

FailureOr<EGraphUnionResult> EGraph::unionValues(EValue lhs, EValue rhs) {
  FailureOr<std::pair<StringAttr, StringAttr>> symbolPair =
      getUnionSymbolPair(lhs, rhs);
  if (failed(symbolPair))
    return failure();

  SmallVector<std::pair<StringAttr, StringAttr>, 1> symbolPairs = {*symbolPair};
  return unionLeaderSymbols(symbolPairs);
}

FailureOr<EGraphUnionResult> EGraph::unionValueTuples(ArrayRef<EValue> lhs,
                                                      ArrayRef<EValue> rhs) {
  if (lhs.size() != rhs.size())
    return failure();

  SmallVector<std::pair<StringAttr, StringAttr>, 4> symbolPairs;
  symbolPairs.reserve(lhs.size());
  for (auto [lhsValue, rhsValue] : llvm::zip_equal(lhs, rhs)) {
    FailureOr<std::pair<StringAttr, StringAttr>> symbolPair =
        getUnionSymbolPair(lhsValue, rhsValue);
    if (failed(symbolPair))
      return failure();
    symbolPairs.push_back(*symbolPair);
  }
  return unionLeaderSymbols(symbolPairs);
}

FailureOr<EGraphRebuildResult> EGraph::rebuild(Operation *root) {
  if (!root)
    return failure();

  LogicalResult result = success();

  EGraphOp egraph = indexedEGraph;
  while (true) {
    result = rewriteSymbolicCandidateRefsToLeaders(egraph);
    if (failed(result))
      break;
    result = rewriteReturnTargetsToLeaders(egraph);
    if (failed(result))
      break;
    result = indexEGraph(egraph);
    if (failed(result))
      break;

    FailureOr<EGraphUnionResult> collisionUnions =
        unionSymbolicStructuralCollisions();
    if (failed(collisionUnions))
      return failure();
    if (!collisionUnions->changed)
      break;
  }
  if (succeeded(result))
    result = materializeSymbolicUnions(egraph);
  if (succeeded(result))
    result = rewriteReturnTargetsToLeaders(egraph);
  if (succeeded(result))
    result = removeDeadSymbolicEClasses(egraph);
  if (succeeded(result))
    result = indexEGraph(egraph);
  if (succeeded(result))
    result = markDuplicateOccurrencesDead();
  if (failed(result))
    return failure();

  auto appendUniqueRef = [](SmallVectorImpl<EOpRefBase> &refs, EOpRefBase ref) {
    for (EOpRefBase existing : refs) {
      if (existing == ref)
        return;
    }
    refs.push_back(ref);
  };

  EGraphRebuildResult rebuildResult;
  for (EOpRefBase ref : getOpRefs())
    appendUniqueRef(rebuildResult.newCandidateRoots, ref);

  for (StringAttr symbolName : touchedSymbolNames) {
    for (EOpRefBase ref : getParentCandidates(getValue(symbolName)))
      appendUniqueRef(rebuildResult.affectedParentCandidates, ref);
  }

  clearTouchedEClasses();
  return rebuildResult;
}

FailureOr<EValue> EGraph::tryResolveValueInOccurrence(
    const EGraphIndex::OperationOccurrence &occurrence, Value value) {
  // Resolve block arguments through the candidate-local symbol mapping first,
  // then resolve intra-candidate results, and finally fall back to indexed
  // egraph values.
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() == occurrence.operation->getBlock()) {
      unsigned argNumber = blockArg.getArgNumber();
      if (argNumber >= occurrence.candidateArgSymbols.size())
        return failure();
      StringAttr symbolName = occurrence.candidateArgSymbols[argNumber];
      if (!symbolName)
        return failure();
      return EValue(this, symbolName);
    }
  }

  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *definingOp = result.getOwner();
    for (const EGraphIndex::OperationOccurrence &candidateOccurrence :
         index.getOccurrences()) {
      if (!candidateOccurrence.live ||
          candidateOccurrence.eclass != occurrence.eclass ||
          candidateOccurrence.candidateOrdinal != occurrence.candidateOrdinal ||
          candidateOccurrence.operation != definingOp)
        continue;

      unsigned resultNumber = result.getResultNumber();
      if (resultNumber >= candidateOccurrence.resultSlotForOpResult.size())
        break;

      if (resultNumber < candidateOccurrence.resultSymbolsForOpResult.size()) {
        StringAttr symbolName =
            candidateOccurrence.resultSymbolsForOpResult[resultNumber];
        if (symbolName)
          return EValue(this, symbolName);
      }
      return failure();
    }
  }

  FailureOr<EValue> lookedUp = lookupValue(value);
  if (succeeded(lookedUp))
    return *lookedUp;

  return failure();
}

EValue EGraph::resolveValueInOccurrence(
    const EGraphIndex::OperationOccurrence &occurrence, Value value) {
  FailureOr<EValue> resolved = tryResolveValueInOccurrence(occurrence, value);
  if (succeeded(resolved))
    return *resolved;

  assert(false && "value is not registered in the egraph index");
  return EValue();
}

FailureOr<EGraphStructuralKey>
EGraph::buildStructuralKey(const EGraphIndex::OperationOccurrence &occurrence) {
  Operation *operation = occurrence.operation;
  if (!operation)
    return failure();

  EGraphStructuralKey key;
  key.operationName = operation->getName();
  key.attributes = operation->getAttrDictionary();

  for (Type resultType : operation->getResultTypes())
    key.resultTypes.push_back(resultType);

  for (Value operand : operation->getOperands()) {
    FailureOr<EValue> child = tryResolveValueInOccurrence(occurrence, operand);
    if (failed(child))
      return failure();

    FailureOr<StringAttr> childLeaderSymbol = getLeaderSymbolForHashing(*child);
    if (failed(childLeaderSymbol))
      return failure();
    key.childLeaderSymbols.push_back(*childLeaderSymbol);
  }

  return key;
}

EGraphHashConsEntry
EGraph::buildHashConsEntry(const EGraphIndex::OperationOccurrence &occurrence) {
  EGraphHashConsEntry entry;
  entry.eclass = occurrence.eclass;
  entry.operation = occurrence.operation;
  entry.resultKeyValues.resize(occurrence.operation->getNumResults());
  entry.resultSymbols.resize(occurrence.operation->getNumResults());

  for (auto indexedSlot : llvm::enumerate(occurrence.resultSlotForOpResult)) {
    unsigned resultSlot = indexedSlot.value();
    if (resultSlot == EGraphIndex::kInvalidResultSlot)
      continue;

    if (indexedSlot.index() < occurrence.resultSymbolsForOpResult.size()) {
      StringAttr resultSymbol =
          occurrence.resultSymbolsForOpResult[indexedSlot.index()];
      if (resultSymbol) {
        entry.resultSymbols[indexedSlot.index()] = resultSymbol;
        continue;
      }
    }
  }

  return entry;
}

LogicalResult EGraph::markDuplicateOccurrencesDead() {
  for (auto indexedOccurrence : llvm::enumerate(index.getOccurrences())) {
    const EGraphIndex::OperationOccurrence &occurrence =
        indexedOccurrence.value();
    if (!occurrence.live)
      continue;

    FailureOr<EGraphStructuralKey> key = buildStructuralKey(occurrence);
    if (failed(key))
      return failure();

    const EGraphHashConsEntry *entry = index.lookupStructuralEntry(*key);
    if (!entry)
      return failure();

    if (entry->operation != occurrence.operation)
      index.markOccurrenceDead(indexedOccurrence.index());
  }
  return success();
}

bool EGraph::isSymbolicAliasCandidate(EClassOp eclass,
                                      unsigned candidateOrdinal) const {
  SmallVector<FlatSymbolRefAttr> refs;
  if (failed(readCandidateSymbolRefs(eclass, candidateOrdinal, refs)) ||
      refs.size() != 1)
    return false;

  auto candidateIt = eclass.getCandidates().begin();
  for (unsigned i = 0; i < candidateOrdinal; ++i)
    ++candidateIt;
  Region &candidate = *candidateIt;
  if (candidate.empty() || !llvm::hasSingleElement(candidate))
    return false;

  Block &block = candidate.front();
  if (block.getNumArguments() != 1)
    return false;

  auto yield = dyn_cast_or_null<YieldOp>(block.getTerminator());
  return yield && yield.getNumOperands() == 1 &&
         yield.getOperand(0) == block.getArgument(0);
}

StringAttr EGraph::findLeaderSymbol(StringAttr symbolName) const {
  StringAttr root = symbolName;
  while (root) {
    auto it = symbolicUnionParents.find(root);
    if (it == symbolicUnionParents.end() || it->second == root)
      return root;
    root = it->second;
  }
  return symbolName;
}

FailureOr<EClassOp>
EGraph::recreateSymbolicEClass(EClassOp eclass, ArrayAttr candidateRefs,
                               ArrayRef<EClassOp> absorbedMembers,
                               bool preserveExistingCandidates,
                               unsigned extraCandidateCount) {
  if (!eclass || !candidateRefs)
    return failure();

  OpBuilder builder(eclass.getContext());
  SmallVector<Attribute> rebuiltCandidateRefs(candidateRefs.begin(),
                                              candidateRefs.end());
  unsigned absorbedCandidateCount = 0;
  for (EClassOp absorbedMember : absorbedMembers) {
    if (!absorbedMember)
      return failure();
    ArrayAttr memberRefs = absorbedMember.getCandidateRefs();
    rebuiltCandidateRefs.append(memberRefs.begin(), memberRefs.end());
    absorbedCandidateCount += absorbedMember.getCandidates().size();
  }

  StringAttr eclassName = eclass.getSymNameAttr();
  unsigned preservedCandidateCount =
      preserveExistingCandidates ? eclass.getCandidates().size() : 0;
  auto newEClass = EClassOp::create(
      builder, eclass.getLoc(), eclass.getSymNameAttr(),
      eclass.getPayloadTypeAttr(), builder.getArrayAttr(rebuiltCandidateRefs),
      preservedCandidateCount + absorbedCandidateCount + extraCandidateCount);
  Operation *newOp = newEClass.getOperation();

  for (NamedAttribute attr : eclass->getAttrs()) {
    StringRef attrName = attr.getName().getValue();
    if (attrName == mlir::SymbolTable::getSymbolAttrName() ||
        attrName == "payload_type" || attrName == "candidate_refs")
      continue;
    newOp->setAttr(attr.getName(), attr.getValue());
  }

  auto newCandidateIt = newEClass.getCandidates().begin();
  auto cloneCandidate = [&](Region &candidate) {
    Block &oldBlock = candidate.front();
    Region &newCandidate = *newCandidateIt;
    IRMapping mapper;
    Block &newBlock = newCandidate.emplaceBlock();
    for (BlockArgument oldArg : oldBlock.getArguments())
      mapper.map(oldArg,
                 newBlock.addArgument(oldArg.getType(), oldArg.getLoc()));

    OpBuilder bodyBuilder(&newBlock, newBlock.begin());
    for (Operation &operation : oldBlock)
      bodyBuilder.insert(operation.clone(mapper));
    ++newCandidateIt;
  };

  if (preserveExistingCandidates) {
    for (Region &candidate : eclass.getCandidates())
      cloneCandidate(candidate);
  }
  for (EClassOp absorbedMember : absorbedMembers)
    for (Region &candidate : absorbedMember.getCandidates())
      cloneCandidate(candidate);

  eclass->getBlock()->getOperations().insert(eclass->getIterator(), newOp);
  payloadSymbols[eclassName] = newOp;
  eclass->dropAllDefinedValueUses();
  eclass->dropAllReferences();
  eclass.erase();
  return newEClass;
}

FailureOr<bool>
EGraph::hasSymbolicAliasCandidate(EClassOp eclass,
                                  StringAttr memberSymbol) const {
  if (!eclass || !memberSymbol)
    return failure();

  ArrayAttr candidateRefs = eclass.getCandidateRefs();
  for (auto indexedRegion : llvm::enumerate(eclass.getCandidates())) {
    unsigned candidateOrdinal = indexedRegion.index();
    if (candidateOrdinal >= candidateRefs.size())
      return eclass.emitOpError("missing candidate_refs entry #")
             << candidateOrdinal;

    auto row = dyn_cast<ArrayAttr>(candidateRefs[candidateOrdinal]);
    if (!row)
      return eclass.emitOpError("candidate_refs entry #")
             << candidateOrdinal << " must be a symbol ref array attribute";
    if (row.size() != 1)
      continue;

    auto ref = dyn_cast<FlatSymbolRefAttr>(row[0]);
    if (!ref || ref.getAttr() != memberSymbol)
      continue;
    if (isSymbolicAliasCandidate(eclass, candidateOrdinal))
      return true;
  }

  return false;
}

LogicalResult EGraph::rewriteSymbolicCandidateRefsToLeaders(EGraphOp egraph) {
  SmallVector<StringAttr, 8> eclassNames;
  for (Operation &op : egraph.getBody().front()) {
    auto eclass = dyn_cast<EClassOp>(op);
    if (eclass)
      eclassNames.push_back(eclass.getSymNameAttr());
  }

  for (StringAttr eclassName : eclassNames) {
    auto eclass = dyn_cast_or_null<EClassOp>(lookupPayloadSymbol(eclassName));
    if (!eclass)
      continue;

    OpBuilder builder(eclass.getContext());
    SmallVector<Attribute> rewrittenRows;
    rewrittenRows.reserve(eclass.getCandidates().size());
    bool changed = false;
    for (auto indexedRegion : llvm::enumerate(eclass.getCandidates())) {
      unsigned candidateOrdinal = indexedRegion.index();
      SmallVector<FlatSymbolRefAttr> refs;
      if (failed(readCandidateSymbolRefs(eclass, candidateOrdinal, refs)))
        return failure();

      SmallVector<Attribute> rewrittenRefs;
      rewrittenRefs.reserve(refs.size());
      bool preserveMemberRefs =
          isSymbolicAliasCandidate(eclass, candidateOrdinal);
      for (FlatSymbolRefAttr ref : refs) {
        StringAttr rewrittenSymbol = preserveMemberRefs
                                         ? ref.getAttr()
                                         : findLeaderSymbol(ref.getAttr());
        if (!rewrittenSymbol)
          return failure();
        changed |= rewrittenSymbol != ref.getAttr();
        rewrittenRefs.push_back(FlatSymbolRefAttr::get(rewrittenSymbol));
      }
      rewrittenRows.push_back(builder.getArrayAttr(rewrittenRefs));
    }

    if (!changed)
      continue;

    if (failed(recreateSymbolicEClass(eclass,
                                      builder.getArrayAttr(rewrittenRows))))
      return failure();
  }

  return success();
}

LogicalResult EGraph::materializeSymbolicAlias(EClassOp leader,
                                               StringAttr memberSymbol) {
  if (!leader || !memberSymbol || leader.getSymNameAttr() == memberSymbol)
    return success();

  Operation *memberOp = lookupPayloadSymbol(memberSymbol);
  if (auto input = dyn_cast_or_null<InputOp>(memberOp))
    return appendSymbolicAliasCandidate(leader, memberSymbol);

  auto member = dyn_cast_or_null<EClassOp>(memberOp);
  if (!member || member.getCandidates().empty())
    return success();

  if (failed(
          recreateSymbolicEClass(leader, leader.getCandidateRefs(), {member})))
    return failure();

  OpBuilder builder(member.getContext());
  if (failed(recreateSymbolicEClass(member, builder.getArrayAttr({}),
                                    /*absorbedMembers=*/{},
                                    /*preserveExistingCandidates=*/false)))
    return failure();
  return success();
}

LogicalResult EGraph::appendSymbolicAliasCandidate(EClassOp leader,
                                                   StringAttr memberSymbol) {
  if (!leader || !memberSymbol || leader.getSymNameAttr() == memberSymbol)
    return success();

  Operation *memberOp = lookupPayloadSymbol(memberSymbol);
  auto input = dyn_cast_or_null<InputOp>(memberOp);
  if (!input)
    return success();

  Type memberType = input.getPayloadType();
  if (!memberType || memberType != leader.getPayloadType())
    return failure();

  FailureOr<bool> hasAlias = hasSymbolicAliasCandidate(leader, memberSymbol);
  if (failed(hasAlias))
    return failure();
  if (*hasAlias)
    return success();

  OpBuilder builder(leader.getContext());
  SmallVector<Attribute, 4> candidateRefs(leader.getCandidateRefs().begin(),
                                          leader.getCandidateRefs().end());
  candidateRefs.push_back(
      builder.getArrayAttr({FlatSymbolRefAttr::get(memberSymbol)}));

  FailureOr<EClassOp> rebuilt = recreateSymbolicEClass(
      leader, builder.getArrayAttr(candidateRefs), {}, true, 1);
  if (failed(rebuilt))
    return failure();

  Region &candidate = rebuilt->getCandidates().back();
  Block &block = candidate.emplaceBlock();
  Value aliasArg = block.addArgument(memberType, rebuilt->getLoc());
  OpBuilder bodyBuilder(&block, block.begin());
  YieldOp::create(bodyBuilder, rebuilt->getLoc(), aliasArg);
  return success();
}

LogicalResult EGraph::removeDeadSymbolicEClasses(EGraphOp egraph) {
  if (!egraph || egraph.isExternal())
    return failure();

  SmallVector<EClassOp, 4> deadEClasses;
  for (Operation &op : egraph.getBody().front()) {
    auto eclass = dyn_cast<EClassOp>(op);
    if (!eclass || !eclass.getCandidates().empty())
      continue;

    auto uses = SymbolTable::getSymbolUses(eclass.getOperation(),
                                           egraph.getOperation());
    if (!uses)
      return eclass.emitOpError("could not analyze dead eclass symbol uses");
    // Empty eclasses referenced by preserved alias candidates are not dead.
    if (uses->begin() != uses->end())
      continue;

    deadEClasses.push_back(eclass);
  }

  for (EClassOp eclass : llvm::reverse(deadEClasses)) {
    payloadSymbols.erase(eclass.getSymNameAttr());
    eclass.erase();
  }

  return success();
}

LogicalResult EGraph::materializeSymbolicUnions(EGraphOp egraph) {
  SmallVector<StringAttr, 8> symbolNames;
  for (Operation &op : egraph.getBody().front()) {
    auto symbolOp = dyn_cast<SymbolOpInterface>(&op);
    if (!symbolOp || !symbolOp.getNameAttr())
      continue;
    symbolNames.push_back(symbolOp.getNameAttr());
  }

  for (StringAttr symbolName : symbolNames) {
    StringAttr leaderName = findLeaderSymbol(symbolName);
    if (!leaderName || leaderName == symbolName)
      continue;

    auto leader = dyn_cast_or_null<EClassOp>(lookupPayloadSymbol(leaderName));
    if (!leader)
      continue;

    if (failed(materializeSymbolicAlias(leader, symbolName)))
      return failure();
  }

  return success();
}

LogicalResult EGraph::rewriteReturnTargetsToLeaders(EGraphOp egraph) {
  auto returnOp =
      dyn_cast_or_null<ReturnOp>(egraph.getBody().front().getTerminator());
  if (!returnOp)
    return failure();

  Builder builder(egraph.getContext());
  SmallVector<Attribute> updatedTargets;
  updatedTargets.reserve(returnOp.getTargets().size());
  bool changed = false;
  for (Attribute attr : returnOp.getTargets()) {
    auto target = cast<FlatSymbolRefAttr>(attr);
    StringAttr leaderName = findLeaderSymbol(target.getAttr());
    if (!leaderName)
      return failure();
    changed |= leaderName != target.getAttr();
    updatedTargets.push_back(FlatSymbolRefAttr::get(leaderName));
  }

  if (changed)
    returnOp->setAttr("targets", builder.getArrayAttr(updatedTargets));
  return success();
}

FailureOr<EGraphUnionResult> EGraph::unionSymbolicStructuralCollisions() {
  SmallVector<std::pair<StringAttr, StringAttr>, 4> symbolPairs;
  for (const EGraphIndex::OperationOccurrence &occurrence :
       index.getOccurrences()) {
    if (!occurrence.live)
      continue;

    FailureOr<EGraphStructuralKey> key = buildStructuralKey(occurrence);
    if (failed(key))
      return failure();

    const EGraphHashConsEntry *entry = index.lookupStructuralEntry(*key);
    if (!entry || !entry->eclass || entry->operation == occurrence.operation)
      continue;

    EClassOp entryEClass = entry->eclass;
    EClassOp occurrenceEClass = occurrence.eclass;
    FailureOr<std::pair<StringAttr, StringAttr>> symbolPair =
        getUnionSymbolPair(getValue(entryEClass.getSymNameAttr()),
                           getValue(occurrenceEClass.getSymNameAttr()));
    if (failed(symbolPair))
      return failure();
    symbolPairs.push_back(*symbolPair);
  }

  return unionLeaderSymbols(symbolPairs);
}

Value EGraph::findLeaderKey(Value keyValue) const {
  Value root = keyValue;
  while (root) {
    auto it = unionParents.find(root);
    if (it == unionParents.end() || it->second == root)
      return root;
    root = it->second;
  }
  return keyValue;
}

FailureOr<std::pair<StringAttr, StringAttr>>
EGraph::getUnionSymbolPair(EValue lhs, EValue rhs) const {
  auto getOwnedLeaderSymbol = [&](EValue value) -> FailureOr<StringAttr> {
    if (!value || value.getGraph() != this || !value.getSymbolNameAttr())
      return failure();
    if (value.getResultIndex() != 0)
      return failure();

    Operation *symbolOp = lookupPayloadSymbol(value.getSymbolNameAttr());
    if (!symbolOp || !getPayloadTypeForSymbol(symbolOp))
      return failure();

    return findLeaderSymbol(value.getSymbolNameAttr());
  };

  FailureOr<StringAttr> lhsSymbol = getOwnedLeaderSymbol(lhs);
  FailureOr<StringAttr> rhsSymbol = getOwnedLeaderSymbol(rhs);
  if (failed(lhsSymbol) || failed(rhsSymbol))
    return failure();

  Operation *lhsOp = lookupPayloadSymbol(*lhsSymbol);
  Operation *rhsOp = lookupPayloadSymbol(*rhsSymbol);
  Type lhsType = getPayloadTypeForSymbol(lhsOp);
  Type rhsType = getPayloadTypeForSymbol(rhsOp);
  if (!lhsType || !rhsType || lhsType != rhsType)
    return failure();

  bool lhsIsEClass = isa_and_nonnull<EClassOp>(lhsOp);
  bool rhsIsEClass = isa_and_nonnull<EClassOp>(rhsOp);
  if (!lhsIsEClass && rhsIsEClass)
    return std::make_pair(*rhsSymbol, *lhsSymbol);

  return std::make_pair(*lhsSymbol, *rhsSymbol);
}

FailureOr<std::pair<Value, Value>> EGraph::getUnionKeyPair(EValue lhs,
                                                           EValue rhs) const {
  auto getOwnedLeaderKey = [&](EValue value) -> FailureOr<Value> {
    if (!value || value.getGraph() != this)
      return failure();

    const EGraphIndex::ValueLookup *lookup = index.lookupValue(value.keyValue);
    if (!lookup || lookup->ownership != OperationOwnership::EGraphOwned ||
        !lookup->keyValue)
      return failure();

    return findLeaderKey(lookup->keyValue);
  };

  FailureOr<Value> lhsKey = getOwnedLeaderKey(lhs);
  FailureOr<Value> rhsKey = getOwnedLeaderKey(rhs);
  if (failed(lhsKey) || failed(rhsKey))
    return failure();

  if (failed(validateUnionKeyPair(*lhsKey, *rhsKey)))
    return failure();

  return std::make_pair(*lhsKey, *rhsKey);
}

EGraphUnionResult EGraph::unionLeaderSymbols(
    ArrayRef<std::pair<StringAttr, StringAttr>> symbolPairs) {
  EGraphUnionResult result;
  result.leaderValues.reserve(symbolPairs.size());
  for (const auto &symbolPair : symbolPairs) {
    StringAttr leader = findLeaderSymbol(symbolPair.first);
    StringAttr member = findLeaderSymbol(symbolPair.second);
    result.leaderValues.push_back(getValue(leader));
    if (leader == member)
      continue;

    symbolicUnionParents.try_emplace(leader, leader);
    symbolicUnionParents[member] = leader;
    result.changed = true;
    recordTouchedSymbol(leader, result);
    recordTouchedSymbol(member, result);
  }
  return result;
}

EGraphUnionResult
EGraph::unionLeaderKeyPairs(ArrayRef<std::pair<Value, Value>> keyPairs) {
  EGraphUnionResult result;
  result.leaderValues.reserve(keyPairs.size());
  for (const std::pair<Value, Value> &keyPair : keyPairs) {
    Value lhsRoot = findLeaderKey(keyPair.first);
    Value rhsRoot = findLeaderKey(keyPair.second);
    assert(succeeded(validateUnionKeyPair(lhsRoot, rhsRoot)) &&
           "union key pair was not prevalidated");

    result.leaderValues.push_back(getValueForKey(lhsRoot));
    if (lhsRoot == rhsRoot)
      continue;

    unionParents.try_emplace(lhsRoot, lhsRoot);
    unionParents[rhsRoot] = lhsRoot;
    result.changed = true;
    recordTouchedEClass(lhsRoot, result);
    recordTouchedEClass(rhsRoot, result);
  }
  return result;
}

void EGraph::recordTouchedSymbol(StringAttr symbolName,
                                 EGraphUnionResult &result) {
  if (!llvm::is_contained(touchedSymbolNames, symbolName))
    touchedSymbolNames.push_back(symbolName);

  Operation *symbolOp = lookupPayloadSymbol(symbolName);
  auto eclass = dyn_cast_or_null<EClassOp>(symbolOp);
  if (!eclass)
    return;

  if (!llvm::is_contained(touchedEClasses, eclass))
    touchedEClasses.push_back(eclass);
  if (!llvm::is_contained(result.touchedEClasses, eclass))
    result.touchedEClasses.push_back(eclass);
}

void EGraph::recordTouchedEClass(Value keyValue, EGraphUnionResult &result) {
  auto opResult = dyn_cast<OpResult>(keyValue);
  if (!opResult)
    return;

  auto eclass = dyn_cast<EClassOp>(opResult.getOwner());
  if (!eclass)
    return;

  if (!llvm::is_contained(touchedEClasses, eclass))
    touchedEClasses.push_back(eclass);
  if (!llvm::is_contained(result.touchedEClasses, eclass))
    result.touchedEClasses.push_back(eclass);
}
