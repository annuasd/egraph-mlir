#include "EGraphTest/IR/EGraphTestOps.h"

namespace mlir {
namespace egraph {
namespace test {

::llvm::LogicalResult
OpBOp::fold(FoldAdaptor adaptor,
            ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  (void)adaptor;
  if (!getTag())
    return failure();

  // Strip the optional tag in place and keep the payload result unchanged.
  removeTagAttr();
  return success();
}

} // namespace test
} // namespace egraph
} // namespace mlir

#define GET_OP_CLASSES
#include "EGraphTest/IR/EGraphTestOps.cpp.inc"
