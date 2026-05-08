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

#ifndef TRANSPOSE_EXAMPLE_INPUT
#define TRANSPOSE_EXAMPLE_INPUT "example/transpose/transpose.mlir"
#endif

namespace {

bool isIdentityPermutation(llvm::ArrayRef<int32_t> permutation) {
  for (size_t i = 0; i < permutation.size(); ++i)
    if (permutation[i] != static_cast<int32_t>(i))
      return false;
  return true;
}

mlir::FailureOr<llvm::SmallVector<int32_t, 4>>
invertPermutation(llvm::ArrayRef<int32_t> permutation) {
  llvm::SmallVector<int32_t, 4> inverse(permutation.size(), -1);
  for (int32_t index = 0; index < static_cast<int32_t>(permutation.size());
       ++index) {
    int32_t axis = permutation[index];
    if (axis < 0 || axis >= static_cast<int32_t>(permutation.size()) ||
        inverse[axis] != -1)
      return mlir::failure();
    inverse[axis] = index;
  }

  for (int32_t axis : inverse)
    if (axis == -1)
      return mlir::failure();

  return inverse;
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
getExampleExtractCost(mlir::Operation *op) {
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

// unary(transpose(x, p)) => transpose(unary(x), p).
struct CombineTransposeUnaryPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::tosa::ExpOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::tosa::ExpOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    return root.getOperand(0).matchDef<mlir::tosa::TransposeOp>(
        [&](mlir::egraph::EOpRef<mlir::tosa::TransposeOp> transposeRef) {
          mlir::tosa::TransposeOp transpose = transposeRef.getOp();
          mlir::Value input = transpose.getInput1();
          auto inputType =
              llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
          if (!inputType)
            return mlir::failure();

          mlir::FailureOr<mlir::RankedTensorType> transposedType =
              getPermutedTensorType(inputType, transpose.getPerms());
          if (mlir::failed(transposedType))
            return mlir::failure();

          auto replacement = mlir::tosa::ExpOp::create(rewriter, root.getLoc(),
                                                       inputType, input);
          auto outerTranspose = mlir::tosa::TransposeOp::create(
              rewriter, root.getLoc(), *transposedType,
              replacement->getResult(0), transpose.getPerms());
          return rewriter.replaceOp(root, outerTranspose->getResults());
        });
  }
};

// transpose(unary(x), p) => unary(transpose(x, p)).
struct CombineUnaryTransposePattern final
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

// binary(transpose(x, p), transpose(y, p)) => transpose(binary(x, y), p).
struct CombineBinaryTransposePattern final
    : public mlir::egraph::EGraphPatternFor<mlir::tosa::MaximumOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::tosa::MaximumOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::tosa::MaximumOp binary = root.getOp();
    return root.getOperand(0).matchDef<mlir::tosa::TransposeOp>(
        [&](mlir::egraph::EOpRef<mlir::tosa::TransposeOp> leftRef) {
          return root.getOperand(1).matchDef<mlir::tosa::TransposeOp>(
              [&](mlir::egraph::EOpRef<mlir::tosa::TransposeOp> rightRef) {
                mlir::tosa::TransposeOp leftTranspose = leftRef.getOp();
                mlir::tosa::TransposeOp rightTranspose = rightRef.getOp();
                if (leftTranspose.getPerms() != rightTranspose.getPerms())
                  return mlir::failure();

                mlir::Value leftInput = leftTranspose.getInput1();
                mlir::Value rightInput = rightTranspose.getInput1();
                auto leftType =
                    llvm::dyn_cast<mlir::RankedTensorType>(leftInput.getType());
                auto rightType = llvm::dyn_cast<mlir::RankedTensorType>(
                    rightInput.getType());
                if (!leftType || !rightType || leftType != rightType)
                  return mlir::failure();

                auto innerBinary = mlir::tosa::MaximumOp::create(
                    rewriter, root.getLoc(), leftType, leftInput, rightInput);
                auto replacement = mlir::tosa::TransposeOp::create(
                    rewriter, root.getLoc(), binary.getOutput().getType(),
                    innerBinary->getResult(0), leftTranspose.getPerms());
                return rewriter.replaceOp(root, replacement->getResults());
              });
        });
  }
};

// binary(transpose(x, p), y) => transpose(binary(x, transpose(y, invP)), p).
struct CombineBinaryLeftTransposePattern final
    : public mlir::egraph::EGraphPatternFor<mlir::tosa::MaximumOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::tosa::MaximumOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::tosa::MaximumOp binary = root.getOp();
    return root.getOperand(0).matchDef<mlir::tosa::TransposeOp>(
        [&](mlir::egraph::EOpRef<mlir::tosa::TransposeOp> leftRef) {
          mlir::tosa::TransposeOp leftTranspose = leftRef.getOp();
          mlir::Value leftInput = leftTranspose.getInput1();
          auto leftType =
              llvm::dyn_cast<mlir::RankedTensorType>(leftInput.getType());
          if (!leftType)
            return mlir::failure();

          mlir::FailureOr<llvm::SmallVector<int32_t, 4>> invPerm =
              invertPermutation(leftTranspose.getPerms());
          if (mlir::failed(invPerm))
            return mlir::failure();

          mlir::Value rightInput = root.getOperation()->getOperand(1);
          auto rightInnerType =
              getPermutedTensorType(rightInput.getType(), *invPerm);
          if (mlir::failed(rightInnerType) || *rightInnerType != leftType)
            return mlir::failure();

          auto rightTranspose = mlir::tosa::TransposeOp::create(
              rewriter, root.getLoc(), *rightInnerType, rightInput, *invPerm);
          auto innerBinary = mlir::tosa::MaximumOp::create(
              rewriter, root.getLoc(), leftType, leftInput,
              rightTranspose->getResult(0));
          auto replacement = mlir::tosa::TransposeOp::create(
              rewriter, root.getLoc(), binary.getOutput().getType(),
              innerBinary->getResult(0), leftTranspose.getPerms());
          return rewriter.replaceOp(root, replacement->getResults());
        });
  }
};

// binary(x, transpose(y, p)) => transpose(binary(transpose(x, invP), y), p).
struct CombineBinaryRightTransposePattern final
    : public mlir::egraph::EGraphPatternFor<mlir::tosa::MaximumOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::tosa::MaximumOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::tosa::MaximumOp binary = root.getOp();
    return root.getOperand(1).matchDef<mlir::tosa::TransposeOp>(
        [&](mlir::egraph::EOpRef<mlir::tosa::TransposeOp> rightRef) {
          mlir::tosa::TransposeOp rightTranspose = rightRef.getOp();
          mlir::Value rightInput = rightTranspose.getInput1();
          auto rightType =
              llvm::dyn_cast<mlir::RankedTensorType>(rightInput.getType());
          if (!rightType)
            return mlir::failure();

          mlir::FailureOr<llvm::SmallVector<int32_t, 4>> invPerm =
              invertPermutation(rightTranspose.getPerms());
          if (mlir::failed(invPerm))
            return mlir::failure();

          mlir::Value leftInput = root.getOperation()->getOperand(0);
          auto leftInnerType =
              getPermutedTensorType(leftInput.getType(), *invPerm);
          if (mlir::failed(leftInnerType) || *leftInnerType != rightType)
            return mlir::failure();

          auto leftTranspose = mlir::tosa::TransposeOp::create(
              rewriter, root.getLoc(), *leftInnerType, leftInput, *invPerm);
          auto innerBinary = mlir::tosa::MaximumOp::create(
              rewriter, root.getLoc(), rightType, leftTranspose->getResult(0),
              rightInput);
          auto replacement = mlir::tosa::TransposeOp::create(
              rewriter, root.getLoc(), binary.getOutput().getType(),
              innerBinary->getResult(0), rightTranspose.getPerms());
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

mlir::LogicalResult runExample(mlir::ModuleOp module) {
  auto originalFunc = module.lookupSymbol<mlir::func::FuncOp>("transpose_example");
  if (!originalFunc)
    return module.emitError("expected a func.func named @transpose_example");
  mlir::Block &originalBlock = originalFunc.getBody().front();

  mlir::egraph::EGraphPatternSet patterns;
  patterns.add<CombineBinaryTransposePattern>();
  patterns.add<CombineBinaryLeftTransposePattern>();
  patterns.add<CombineBinaryRightTransposePattern>();
  patterns.add<CombineTransposeBinaryPattern>();
  patterns.add<CombineTransposeUnaryPattern>();
  patterns.add<CombineUnaryTransposePattern>();
  patterns.add<FoldTwoTransposesPattern>();
  patterns.add<FoldNopTransposePattern>();

  if (mlir::failed(mlir::egraph::applyEGraphPatternsAndExtract(
          originalBlock, patterns,
          mlir::egraph::EGraphExtractMode::LinearProgramming,
          getExampleExtractCost)))
    return module.emitError("failed to optimize @transpose_example");

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
      llvm::cl::init(TRANSPOSE_EXAMPLE_INPUT));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR-EGraph TOSA transpose example\n");

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

  return mlir::failed(runExample(*module)) ? 1 : 0;
}
