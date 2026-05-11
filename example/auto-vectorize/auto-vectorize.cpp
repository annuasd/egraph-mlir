#include "MLIREGraph/EGraph/Pattern.h"
#include "MLIREGraph/IR/EGraphDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

#ifndef AUTO_VECTORIZE_EXAMPLE_INPUT
#define AUTO_VECTORIZE_EXAMPLE_INPUT                                           \
  "example/auto-vectorize/auto-vectorize.mlir"
#endif

namespace {

constexpr int64_t kRows = 2;
constexpr int64_t kValueK = 8;
constexpr int64_t kValueN = 4;
constexpr int64_t kPackedK = 2;
constexpr int64_t kLane = 4;

bool hasVectorShape(mlir::Type type, llvm::ArrayRef<int64_t> expectedShape) {
  auto vectorType = llvm::dyn_cast<mlir::VectorType>(type);
  if (!vectorType ||
      vectorType.getRank() != static_cast<int64_t>(expectedShape.size()))
    return false;

  for (auto [actual, expected] :
       llvm::zip_equal(vectorType.getShape(), expectedShape))
    if (actual != expected)
      return false;

  return true;
}

mlir::ArrayAttr getAffineMapArrayAttr(mlir::Builder &builder,
                                      llvm::ArrayRef<mlir::AffineMap> maps) {
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  attrs.reserve(maps.size());
  for (mlir::AffineMap map : maps)
    attrs.push_back(mlir::AffineMapAttr::get(map));
  return builder.getArrayAttr(attrs);
}

mlir::ArrayAttr getIteratorTypeArrayAttr(
    mlir::Builder &builder,
    llvm::ArrayRef<mlir::vector::IteratorType> iteratorTypes) {
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  attrs.reserve(iteratorTypes.size());
  for (mlir::vector::IteratorType iteratorType : iteratorTypes)
    attrs.push_back(mlir::vector::IteratorTypeAttr::get(builder.getContext(),
                                                        iteratorType));
  return builder.getArrayAttr(attrs);
}

mlir::ArrayAttr getPackedKMatmulIndexingMaps(mlir::Builder &builder) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::AffineExpr m = builder.getAffineDimExpr(0);
  mlir::AffineExpr n = builder.getAffineDimExpr(1);
  mlir::AffineExpr ko = builder.getAffineDimExpr(2);
  mlir::AffineExpr lane = builder.getAffineDimExpr(3);

  llvm::SmallVector<mlir::AffineMap, 3> maps = {
      mlir::AffineMap::get(4, 0, {m, ko, lane}, context),
      mlir::AffineMap::get(4, 0, {ko, lane, n}, context),
      mlir::AffineMap::get(4, 0, {m, n}, context),
  };
  return getAffineMapArrayAttr(builder, maps);
}

mlir::ArrayAttr getMatmulIndexingMaps(mlir::Builder &builder) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::AffineExpr m = builder.getAffineDimExpr(0);
  mlir::AffineExpr n = builder.getAffineDimExpr(1);
  mlir::AffineExpr k = builder.getAffineDimExpr(2);

  llvm::SmallVector<mlir::AffineMap, 3> maps = {
      mlir::AffineMap::get(3, 0, {m, k}, context),
      mlir::AffineMap::get(3, 0, {k, n}, context),
      mlir::AffineMap::get(3, 0, {m, n}, context),
  };
  return getAffineMapArrayAttr(builder, maps);
}

mlir::ArrayAttr getMatmulIteratorTypes(mlir::Builder &builder) {
  llvm::SmallVector<mlir::vector::IteratorType, 3> iteratorTypes = {
      mlir::vector::IteratorType::parallel,
      mlir::vector::IteratorType::parallel,
      mlir::vector::IteratorType::reduction,
  };
  return getIteratorTypeArrayAttr(builder, iteratorTypes);
}

mlir::ArrayAttr getPackedKMatmulIteratorTypes(mlir::Builder &builder) {
  llvm::SmallVector<mlir::vector::IteratorType, 4> iteratorTypes = {
      mlir::vector::IteratorType::parallel,
      mlir::vector::IteratorType::parallel,
      mlir::vector::IteratorType::reduction,
      mlir::vector::IteratorType::reduction,
  };
  return getIteratorTypeArrayAttr(builder, iteratorTypes);
}

bool hasStandardMatmulAttrs(mlir::vector::ContractionOp op) {
  mlir::Builder builder(op.getOperation()->getContext());
  return op.getIndexingMaps() == getMatmulIndexingMaps(builder) &&
         op.getIteratorTypes() == getMatmulIteratorTypes(builder) &&
         op.getKind() == mlir::vector::CombiningKind::ADD;
}

bool hasPackedKMatmulAttrs(mlir::vector::ContractionOp op) {
  mlir::Builder builder(op.getOperation()->getContext());
  return op.getIndexingMaps() == getPackedKMatmulIndexingMaps(builder) &&
         op.getIteratorTypes() == getPackedKMatmulIteratorTypes(builder) &&
         op.getKind() == mlir::vector::CombiningKind::ADD;
}

bool hasExpDef(mlir::egraph::EValue value) {
  for (auto def : value.getDefs<mlir::math::ExpOp>()) {
    (void)def;
    return true;
  }
  return false;
}

bool isUnpackedValueMatmul(mlir::vector::ContractionOp op) {
  return hasVectorShape(op.getLhs().getType(), {kRows, kValueK}) &&
         hasVectorShape(op.getRhs().getType(), {kValueK, kValueN}) &&
         hasVectorShape(op.getAcc().getType(), {kRows, kValueN}) &&
         hasVectorShape(op.getOperation()->getResult(0).getType(),
                        {kRows, kValueN}) &&
         hasStandardMatmulAttrs(op);
}

bool isPackedKValueMatmul(mlir::vector::ContractionOp op) {
  return hasVectorShape(op.getLhs().getType(), {kRows, kPackedK, kLane}) &&
         hasVectorShape(op.getRhs().getType(), {kPackedK, kLane, kValueN}) &&
         hasVectorShape(op.getAcc().getType(), {kRows, kValueN}) &&
         hasVectorShape(op.getOperation()->getResult(0).getType(),
                        {kRows, kValueN}) &&
         hasPackedKMatmulAttrs(op);
}

mlir::FailureOr<mlir::egraph::EGraphExtractCost>
getExampleExtractCost(mlir::Operation *op) {
  if (auto contract = llvm::dyn_cast<mlir::vector::ContractionOp>(op)) {
    if (isPackedKValueMatmul(contract))
      return mlir::egraph::EGraphExtractCost(2);
    return mlir::egraph::EGraphExtractCost(8);
  }
  if (auto exp = llvm::dyn_cast<mlir::math::ExpOp>(op)) {
    auto operandType =
        llvm::dyn_cast<mlir::VectorType>(exp.getOperand().getType());
    if (operandType && operandType.getRank() > 2)
      return mlir::egraph::EGraphExtractCost(1);
    return mlir::egraph::EGraphExtractCost(4);
  }
  if (llvm::isa<mlir::vector::ShapeCastOp>(op))
    return mlir::egraph::EGraphExtractCost(1);
  return mlir::egraph::EGraphExtractCost(4);
}

// Rewrite the value matmul into the packed-K contraction form.
struct VectorizeValueMatmulPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::vector::ContractionOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::vector::ContractionOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::vector::ContractionOp contract = root.getOp();
    if (!isUnpackedValueMatmul(contract) || !hasExpDef(root.getOperand(0)))
      return mlir::failure();

    mlir::Type elementType = contract.getLhs().getType().getElementType();
    auto packedLhsType =
        mlir::VectorType::get({kRows, kPackedK, kLane}, elementType);
    auto packedRhsType =
        mlir::VectorType::get({kPackedK, kLane, kValueN}, elementType);

    auto packedLhs = mlir::vector::ShapeCastOp::create(
        rewriter, root.getLoc(), packedLhsType, contract.getLhs());
    auto packedRhs = mlir::vector::ShapeCastOp::create(
        rewriter, root.getLoc(), packedRhsType, contract.getRhs());
    auto replacement = mlir::vector::ContractionOp::create(
        rewriter, root.getLoc(),
        contract.getOperation()->getResult(0).getType(), packedLhs.getResult(),
        packedRhs.getResult(), contract.getAcc(),
        getPackedKMatmulIndexingMaps(rewriter),
        getPackedKMatmulIteratorTypes(rewriter),
        mlir::vector::CombiningKindAttr::get(rewriter.getContext(),
                                             contract.getKind()),
        contract.getFastmathAttr());
    return rewriter.replaceOp(root, replacement->getResults());
  }
};

// shape_cast(exp(x)) => exp(shape_cast(x)).
struct PackExpPropagationPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::vector::ShapeCastOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::vector::ShapeCastOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::vector::ShapeCastOp pack = root.getOp();
    return root.getOperand(0).matchDef<mlir::math::ExpOp>(
        [&](mlir::egraph::EOpRef<mlir::math::ExpOp> expRef) {
          mlir::math::ExpOp exp = expRef.getOp();
          if (!hasVectorShape(exp.getOperand().getType(), {kRows, kValueK}) ||
              !hasVectorShape(pack.getResult().getType(),
                              {kRows, kPackedK, kLane}))
            return mlir::failure();

          auto packedInput = mlir::vector::ShapeCastOp::create(
              rewriter, root.getLoc(), pack.getResult().getType(),
              exp.getOperand());
          auto replacement = mlir::math::ExpOp::create(
              rewriter, root.getLoc(), pack.getResult().getType(),
              packedInput.getResult(), exp.getFastmathAttr());
          return rewriter.replaceOp(root, replacement->getResults());
        });
  }
};

// exp(shape_cast(x)) => shape_cast(exp(x)).
struct UnpackExpPropagationPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::math::ExpOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::math::ExpOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::math::ExpOp exp = root.getOp();
    if (!hasVectorShape(exp.getResult().getType(), {kRows, kPackedK, kLane}))
      return mlir::failure();

    return root.getOperand(0).matchDef<mlir::vector::ShapeCastOp>(
        [&](mlir::egraph::EOpRef<mlir::vector::ShapeCastOp> packRef) {
          mlir::vector::ShapeCastOp pack = packRef.getOp();
          if (!hasVectorShape(pack.getSource().getType(), {kRows, kValueK}) ||
              !hasVectorShape(pack.getResult().getType(),
                              {kRows, kPackedK, kLane}))
            return mlir::failure();

          auto unpackedExp = mlir::math::ExpOp::create(
              rewriter, root.getLoc(), pack.getSource().getType(),
              pack.getSource(), exp.getFastmathAttr());
          auto replacement = mlir::vector::ShapeCastOp::create(
              rewriter, root.getLoc(), exp.getResult().getType(),
              unpackedExp.getResult());
          return rewriter.replaceOp(root, replacement->getResults());
        });
  }
};

// shape_cast(shape_cast(x)) => x when the endpoint type matches.
struct FoldShapeCastPairPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::vector::ShapeCastOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::vector::ShapeCastOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::vector::ShapeCastOp outer = root.getOp();
    return root.getOperand(0).matchDef<mlir::vector::ShapeCastOp>(
        [&](mlir::egraph::EOpRef<mlir::vector::ShapeCastOp> innerRef) {
          mlir::vector::ShapeCastOp inner = innerRef.getOp();
          if (inner.getSource().getType() != outer.getResult().getType())
            return mlir::failure();
          return rewriter.replaceOp(root, {inner.getSource()});
        });
  }
};

mlir::LogicalResult runExample(mlir::ModuleOp module) {
  auto originalFunc =
      module.lookupSymbol<mlir::func::FuncOp>("auto_vectorize_example");
  if (!originalFunc)
    return module.emitError(
        "expected a func.func named @auto_vectorize_example");
  mlir::Block &originalBlock = originalFunc.getBody().front();

  mlir::egraph::EGraphPatternSet patterns;
  patterns.add<VectorizeValueMatmulPattern>();
  patterns.add<PackExpPropagationPattern>();
  patterns.add<UnpackExpPropagationPattern>();
  patterns.add<FoldShapeCastPairPattern>();

  if (mlir::failed(mlir::egraph::applyEGraphPatternsAndExtract(
          originalBlock, patterns,
          mlir::egraph::EGraphExtractMode::LinearProgramming,
          getExampleExtractCost)))
    return module.emitError("failed to optimize @auto_vectorize_example");

  llvm::outs() << "Optimized IR:\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";
  return mlir::success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input mlir>"),
      llvm::cl::init(AUTO_VECTORIZE_EXAMPLE_INPUT));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR-EGraph auto-vectorize example\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::egraph::EGraphDialect,
                  mlir::func::FuncDialect, mlir::math::MathDialect,
                  mlir::vector::VectorDialect>();

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  if (!module) {
    llvm::errs() << "failed to parse " << inputFilename << "\n";
    return 1;
  }

  return mlir::failed(runExample(*module)) ? 1 : 0;
}
