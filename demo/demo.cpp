#include "MLIREGraph/EGraph/Pattern.h"
#include "MLIREGraph/IR/EGraphDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#ifndef MLIR_EGRAPH_DEMO_INPUT
#define MLIR_EGRAPH_DEMO_INPUT "demo/demo.mlir"
#endif

namespace {

bool hasConstantIntegerDef(mlir::egraph::EValue value, int64_t expected) {
  for (auto def :
       value.getDefs<mlir::arith::ConstantOp>()) {
    auto integer = llvm::dyn_cast<mlir::IntegerAttr>(def.getOp().getValue());
    if (integer && integer.getInt() == expected)
      return true;
  }
  return false;
}

mlir::FailureOr<mlir::egraph::EGraphExtractCost>
getDemoExtractCost(mlir::Operation *candidate) {
  llvm::StringRef operationName = candidate->getName().getStringRef();
  if (operationName == mlir::arith::ConstantOp::getOperationName())
    return mlir::egraph::EGraphExtractCost(1);
  if (operationName == mlir::arith::ShLIOp::getOperationName())
    return mlir::egraph::EGraphExtractCost(1);
  if (operationName == mlir::arith::MulIOp::getOperationName())
    return mlir::egraph::EGraphExtractCost(4);
  if (operationName == mlir::arith::DivSIOp::getOperationName())
    return mlir::egraph::EGraphExtractCost(8);
  return mlir::egraph::EGraphExtractCost(16);
}

struct MulByTwoToShiftPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::arith::MulIOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::arith::MulIOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    auto resultType = llvm::dyn_cast<mlir::IntegerType>(
        root.getOperation()->getResult(0).getType());
    if (!resultType)
      return mlir::failure();

    mlir::Value shiftedPayload;
    if (hasConstantIntegerDef(root.getOperand(0), 2)) {
      shiftedPayload = root.getOperation()->getOperand(1);
    } else if (hasConstantIntegerDef(root.getOperand(1), 2)) {
      shiftedPayload = root.getOperation()->getOperand(0);
    } else {
      return mlir::failure();
    }

    auto one = mlir::arith::ConstantOp::create(
        rewriter, root.getLoc(), rewriter.getIntegerAttr(resultType, 1));
    auto replacement = mlir::arith::ShLIOp::create(
        rewriter, root.getLoc(), shiftedPayload, one.getResult());
    return rewriter.replaceOp(root, replacement->getResults());
  }
};

struct ReassociateDivPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::arith::DivSIOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::arith::DivSIOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    return root.getOperand(0).matchDef<mlir::arith::MulIOp>(
        [&](mlir::egraph::EOpRef<mlir::arith::MulIOp> mul) {
          auto rotatedDiv = mlir::arith::DivSIOp::create(
              rewriter, root.getLoc(), mul.getOperation()->getOperand(1),
              root.getOperation()->getOperand(1));
          auto replacement = mlir::arith::MulIOp::create(
              rewriter, root.getLoc(), mul.getOperation()->getOperand(0),
              rotatedDiv.getResult());
          return rewriter.replaceOp(root, replacement->getResults());
        });
  }
};

struct DivSelfToOnePattern final
    : public mlir::egraph::EGraphPatternFor<mlir::arith::DivSIOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::arith::DivSIOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    auto resultType = llvm::dyn_cast<mlir::IntegerType>(
        root.getOperation()->getResult(0).getType());
    if (!resultType || !root.getOperand(0).isEquivalentTo(root.getOperand(1)))
      return mlir::failure();

    auto one = mlir::arith::ConstantOp::create(
        rewriter, root.getLoc(), rewriter.getIntegerAttr(resultType, 1));
    return rewriter.replaceOp(root, one->getResults());
  }
};

struct MulByOneToAliasPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::arith::MulIOp> {
  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::arith::MulIOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::Value aliasedPayload;
    if (hasConstantIntegerDef(root.getOperand(0), 1)) {
      aliasedPayload = root.getOperation()->getOperand(1);
    } else if (hasConstantIntegerDef(root.getOperand(1), 1)) {
      aliasedPayload = root.getOperation()->getOperand(0);
    } else {
      return mlir::failure();
    }
    return rewriter.replaceOp(root, {aliasedPayload});
  }
};

mlir::LogicalResult runDemo(mlir::ModuleOp module) {
  auto originalFunc = module.lookupSymbol<mlir::func::FuncOp>("arith_demo");
  if (!originalFunc)
    return module.emitError("expected a func.func named @arith_demo");
  mlir::Block &originalBlock = originalFunc.getBody().front();

  mlir::egraph::EGraphPatternSet patterns;
  patterns.add<MulByTwoToShiftPattern>();
  patterns.add<ReassociateDivPattern>();
  patterns.add<DivSelfToOnePattern>();
  patterns.add<MulByOneToAliasPattern>();

  if (mlir::failed(mlir::egraph::applyEGraphPatternsAndExtract(
          originalBlock, patterns, mlir::egraph::EGraphExtractMode::Greedy,
          getDemoExtractCost)))
    return module.emitError("failed to optimize @arith_demo");

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
      llvm::cl::init(MLIR_EGRAPH_DEMO_INPUT));
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR-EGraph API demo\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::egraph::EGraphDialect,
                  mlir::func::FuncDialect>();

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  if (!module) {
    llvm::errs() << "failed to parse " << inputFilename << "\n";
    return 1;
  }

  return mlir::failed(runDemo(*module)) ? 1 : 0;
}
