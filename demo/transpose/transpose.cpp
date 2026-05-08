#include "MLIREGraph/EGraph/Pattern.h"
#include "MLIREGraph/IR/EGraphDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
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

#ifndef TRANSPOSE_DEMO_INPUT
#define TRANSPOSE_DEMO_INPUT "demo/transpose/transpose.mlir"
#endif

namespace {

bool isIdentityPermutation(llvm::ArrayRef<int32_t> permutation) {
  for (size_t i = 0; i < permutation.size(); ++i)
    if (permutation[i] != static_cast<int32_t>(i))
      return false;
  return true;
}

mlir::FailureOr<mlir::RankedTensorType>
getPermutedTensorType(mlir::Type type, llvm::ArrayRef<int32_t> permutation) {
  auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!ranked)
    return mlir::failure();

  int64_t rank = ranked.getRank();
  if (rank != static_cast<int64_t>(permutation.size()))
    return mlir::failure();

  llvm::SmallVector<int64_t, 4> shape;
  shape.reserve(permutation.size());
  for (int32_t axis : permutation) {
    if (axis < 0 || axis >= rank)
      return mlir::failure();
    shape.push_back(ranked.getShape()[axis]);
  }
  return mlir::RankedTensorType::get(shape, ranked.getElementType(),
                                     ranked.getEncoding());
}

mlir::FailureOr<llvm::SmallVector<int32_t, 4>>
composePermutations(llvm::ArrayRef<int32_t> first,
                    llvm::ArrayRef<int32_t> second) {
  if (first.size() != second.size())
    return mlir::failure();

  llvm::SmallVector<int32_t, 4> composed;
  composed.reserve(first.size());
  for (int32_t axis : second) {
    if (axis < 0 || axis >= static_cast<int32_t>(first.size()))
      return mlir::failure();
    composed.push_back(first[axis]);
  }
  return composed;
}

mlir::FailureOr<mlir::egraph::EGraphExtractCost>
getDemoExtractCost(mlir::Operation *op) {
  if (llvm::isa<mlir::tosa::TransposeOp>(op))
    return mlir::egraph::EGraphExtractCost(16);
  if (llvm::isa<mlir::tosa::MaximumOp>(op))
    return mlir::egraph::EGraphExtractCost(1);
  if (llvm::isa<mlir::tosa::ExpOp>(op))
    return mlir::egraph::EGraphExtractCost(1);
  return mlir::egraph::EGraphExtractCost(4);
}

struct FoldNopTransposePattern final
    : public mlir::egraph::EGraphPatternFor<mlir::tosa::TransposeOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::tosa::TransposeOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::tosa::TransposeOp transpose = root.getOp();
    if (!isIdentityPermutation(transpose.getPerms()))
      return mlir::failure();

    return rewriter.replaceOp(root, {transpose.getInput1()});
  }
};

// Fold transpose(transpose(x, p1), p2) by composing the permutations.
struct FoldTwoTransposesPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::tosa::TransposeOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::tosa::TransposeOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    return root.getOperand(0).matchDef<mlir::tosa::TransposeOp>(
        [&](mlir::egraph::EOpRef<mlir::tosa::TransposeOp> innerRef) {
          mlir::tosa::TransposeOp outer = root.getOp();
          mlir::tosa::TransposeOp inner = innerRef.getOp();
          mlir::FailureOr<llvm::SmallVector<int32_t, 4>> composed =
              composePermutations(inner.getPerms(), outer.getPerms());
          if (mlir::failed(composed))
            return mlir::failure();

          mlir::Value input = inner.getInput1();
          if (isIdentityPermutation(*composed))
            return rewriter.replaceOp(root, {input});

          auto replacement = mlir::tosa::TransposeOp::create(
              rewriter, root.getLoc(), outer.getOutput().getType(), input,
              *composed);
          return rewriter.replaceOp(root, replacement->getResults());
        });
  }
};

// transpose(unary(x), p) => unary(transpose(x, p)).
struct CombineTransposeUnaryPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::tosa::TransposeOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::tosa::TransposeOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::tosa::TransposeOp transpose = root.getOp();
    return root.getOperand(0).matchDef<mlir::tosa::ExpOp>(
        [&](mlir::egraph::EOpRef<mlir::tosa::ExpOp> unaryRef) {
          mlir::Value input = unaryRef.getOperation()->getOperand(0);
          mlir::FailureOr<mlir::RankedTensorType> transposedType =
              getPermutedTensorType(input.getType(), transpose.getPerms());
          if (mlir::failed(transposedType))
            return mlir::failure();

          auto innerTranspose = mlir::tosa::TransposeOp::create(
              rewriter, root.getLoc(), *transposedType, input,
              transpose.getPerms());
          auto replacement = mlir::tosa::ExpOp::create(
              rewriter, root.getLoc(), *transposedType,
              innerTranspose->getResult(0));
          return rewriter.replaceOp(root, replacement->getResults());
        });
  }
};

// transpose(binary(x, y), p) => binary(transpose(x, p), transpose(y, p)).
struct CombineTransposeBinaryPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::tosa::TransposeOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::tosa::TransposeOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::tosa::TransposeOp transpose = root.getOp();
    return root.getOperand(0).matchDef<mlir::tosa::MaximumOp>(
        [&](mlir::egraph::EOpRef<mlir::tosa::MaximumOp> binaryRef) {
          mlir::Value leftInput = binaryRef.getOperation()->getOperand(0);
          mlir::Value rightInput = binaryRef.getOperation()->getOperand(1);
          auto leftType =
              getPermutedTensorType(leftInput.getType(), transpose.getPerms());
          auto rightType =
              getPermutedTensorType(rightInput.getType(), transpose.getPerms());
          if (mlir::failed(leftType) || mlir::failed(rightType) ||
              *leftType != *rightType)
            return mlir::failure();

          auto leftTranspose = mlir::tosa::TransposeOp::create(
              rewriter, root.getLoc(), *leftType, leftInput,
              transpose.getPerms());
          auto rightTranspose = mlir::tosa::TransposeOp::create(
              rewriter, root.getLoc(), *rightType, rightInput,
              transpose.getPerms());
          auto replacement = mlir::tosa::MaximumOp::create(
              rewriter, root.getLoc(), *leftType, leftTranspose->getResult(0),
              rightTranspose->getResult(0));
          return rewriter.replaceOp(root, replacement->getResults());
        });
  }
};

mlir::LogicalResult runDemo(mlir::ModuleOp module) {
  auto originalFunc = module.lookupSymbol<mlir::func::FuncOp>("transpose_demo");
  if (!originalFunc)
    return module.emitError("expected a func.func named @transpose_demo");
  mlir::Block &originalBlock = originalFunc.getBody().front();

  mlir::egraph::EGraphPatternSet patterns;
  patterns.add<CombineTransposeBinaryPattern>();
  patterns.add<CombineTransposeUnaryPattern>();
  patterns.add<FoldTwoTransposesPattern>();
  patterns.add<FoldNopTransposePattern>();

  if (mlir::failed(mlir::egraph::applyEGraphPatternsAndExtract(
          originalBlock, patterns,
          mlir::egraph::EGraphExtractMode::LinearProgramming,
          getDemoExtractCost)))
    return module.emitError("failed to optimize @transpose_demo");

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
      llvm::cl::init(TRANSPOSE_DEMO_INPUT));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR-EGraph TOSA transpose demo\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::egraph::EGraphDialect, mlir::func::FuncDialect,
                  mlir::tosa::TosaDialect>();

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  if (!module) {
    llvm::errs() << "failed to parse " << inputFilename << "\n";
    return 1;
  }

  return mlir::failed(runDemo(*module)) ? 1 : 0;
}
