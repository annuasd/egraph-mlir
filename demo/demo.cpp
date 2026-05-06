#include "MLIREGraph/EGraph/EGraph.h"
#include "MLIREGraph/EGraph/Extract.h"
#include "MLIREGraph/EGraph/FuncToEGraph.h"
#include "MLIREGraph/EGraph/Pattern.h"
#include "MLIREGraph/IR/EGraphDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#ifndef MLIR_EGRAPH_DEMO_INPUT
#define MLIR_EGRAPH_DEMO_INPUT "demo/demo.mlir"
#endif

namespace {

bool hasConstantIntegerDef(mlir::egraph::EValue value, int64_t expected) {
  for (mlir::egraph::EOpRef<mlir::arith::ConstantOp> def :
       value.getDefs<mlir::arith::ConstantOp>()) {
    auto integer = llvm::dyn_cast<mlir::IntegerAttr>(def.getOp().getValue());
    if (integer && integer.getInt() == expected)
      return true;
  }
  return false;
}

bool hasShiftLeftByOneDef(mlir::egraph::EValue value,
                          mlir::egraph::EValue shiftedValue) {
  for (mlir::egraph::EOpRef<mlir::arith::ShLIOp> def :
       value.getDefs<mlir::arith::ShLIOp>()) {
    if (!def.getOperand(0).isEquivalentTo(shiftedValue) ||
        !hasConstantIntegerDef(def.getOperand(1), 1))
      continue;
    return true;
  }
  return false;
}

bool hasExactDivDef(mlir::egraph::EValue value, mlir::egraph::EValue lhs,
                    mlir::egraph::EValue rhs) {
  for (mlir::egraph::EOpRef<mlir::arith::DivSIOp> def :
       value.getDefs<mlir::arith::DivSIOp>()) {
    if (def.getOperand(0).isEquivalentTo(lhs) &&
        def.getOperand(1).isEquivalentTo(rhs))
      return true;
  }
  return false;
}

bool hasMulOfDivDef(mlir::egraph::EValue value, mlir::egraph::EValue mulLhs,
                    mlir::egraph::EValue divLhs, mlir::egraph::EValue divRhs) {
  for (mlir::egraph::EOpRef<mlir::arith::MulIOp> def :
       value.getDefs<mlir::arith::MulIOp>()) {
    if (!def.getOperand(0).isEquivalentTo(mulLhs) ||
        !hasExactDivDef(def.getOperand(1), divLhs, divRhs))
      continue;
    return true;
  }
  return false;
}

mlir::FailureOr<mlir::egraph::EGraphExtractCost>
getDemoExtractCost(mlir::egraph::EOpRefBase candidate) {
  llvm::StringRef operationName = candidate.getOperationName();
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

mlir::LogicalResult rewriteBlockWithExtractedGraph(
    mlir::Block &block, mlir::egraph::EGraph &graph,
    mlir::egraph::EGraphOp egraph,
    const mlir::egraph::EGraphExtractResult &selection) {
  mlir::OpBuilder builder(block.getParentOp()->getContext());
  block.clear();
  builder.setInsertionPointToEnd(&block);

  llvm::SmallVector<mlir::Value, 4> inputValues(block.getArguments().begin(),
                                                block.getArguments().end());
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> roots =
      mlir::egraph::materializeEGraphExtractResult(graph, egraph, selection,
                                                   builder, inputValues);
  if (mlir::failed(roots))
    return egraph.emitOpError("failed to materialize extracted roots");

  mlir::func::ReturnOp::create(builder, egraph.getLoc(), *roots);
  return mlir::success();
}

mlir::FailureOr<mlir::egraph::EGraphExtractResult>
extractDemoGraph(mlir::egraph::EGraph &graph, mlir::egraph::EGraphOp egraph) {
  mlir::FailureOr<mlir::egraph::EGraphExtractRequest> request =
      mlir::egraph::buildEGraphExtractRequest(graph, egraph);
  if (mlir::failed(request))
    return egraph.emitOpError("failed to build graph extract request");

  return mlir::egraph::extractEGraphGreedily(graph, egraph, *request,
                                             getDemoExtractCost);
}

struct MulByTwoToShiftPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::arith::MulIOp> {
  explicit MulByTwoToShiftPattern(unsigned &matchCount)
      : matchCount(matchCount) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::arith::MulIOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    auto resultType = llvm::dyn_cast<mlir::IntegerType>(
        root.getOperation()->getResult(0).getType());
    if (!resultType)
      return mlir::failure();

    mlir::Value shiftedPayload;
    mlir::egraph::EValue shiftedValue;
    if (hasConstantIntegerDef(root.getOperand(0), 2)) {
      shiftedPayload = root.getOperation()->getOperand(1);
      shiftedValue = root.getOperand(1);
    } else if (hasConstantIntegerDef(root.getOperand(1), 2)) {
      shiftedPayload = root.getOperation()->getOperand(0);
      shiftedValue = root.getOperand(0);
    } else {
      return mlir::failure();
    }

    if (hasShiftLeftByOneDef(root.getResult(0), shiftedValue))
      return mlir::failure();

    auto one = mlir::arith::ConstantOp::create(
        rewriter, root.getLoc(), rewriter.getIntegerAttr(resultType, 1));
    auto replacement = mlir::arith::ShLIOp::create(
        rewriter, root.getLoc(), shiftedPayload, one.getResult());
    ++matchCount;
    return rewriter.replaceOp(root, replacement->getResults());
  }

private:
  unsigned &matchCount;
};

struct ReassociateDivPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::arith::DivSIOp> {
  explicit ReassociateDivPattern(unsigned &matchCount)
      : matchCount(matchCount) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::arith::DivSIOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    return root.getOperand(0).matchDef<mlir::arith::MulIOp>(
        [&](mlir::egraph::EOpRef<mlir::arith::MulIOp> mul) {
          if (hasMulOfDivDef(root.getResult(0), mul.getOperand(0),
                             mul.getOperand(1), root.getOperand(1)))
            return mlir::failure();

          auto rotatedDiv = mlir::arith::DivSIOp::create(
              rewriter, root.getLoc(), mul.getOperation()->getOperand(1),
              root.getOperation()->getOperand(1));
          auto replacement = mlir::arith::MulIOp::create(
              rewriter, root.getLoc(), mul.getOperation()->getOperand(0),
              rotatedDiv.getResult());
          ++matchCount;
          return rewriter.replaceOp(root, replacement->getResults());
        });
  }

private:
  unsigned &matchCount;
};

struct DivSelfToOnePattern final
    : public mlir::egraph::EGraphPatternFor<mlir::arith::DivSIOp> {
  explicit DivSelfToOnePattern(unsigned &matchCount) : matchCount(matchCount) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::arith::DivSIOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    auto resultType = llvm::dyn_cast<mlir::IntegerType>(
        root.getOperation()->getResult(0).getType());
    if (!resultType || !root.getOperand(0).isEquivalentTo(root.getOperand(1)) ||
        hasConstantIntegerDef(root.getResult(0), 1))
      return mlir::failure();

    auto one = mlir::arith::ConstantOp::create(
        rewriter, root.getLoc(), rewriter.getIntegerAttr(resultType, 1));
    ++matchCount;
    return rewriter.replaceOp(root, one->getResults());
  }

private:
  unsigned &matchCount;
};

struct MulByOneToAliasPattern final
    : public mlir::egraph::EGraphPatternFor<mlir::arith::MulIOp> {
  explicit MulByOneToAliasPattern(unsigned &matchCount)
      : matchCount(matchCount) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::egraph::EOpRef<mlir::arith::MulIOp> root,
                  mlir::egraph::EGraphPatternRewriter &rewriter) const final {
    mlir::Value aliasedPayload;
    mlir::egraph::EValue aliasedValue;
    if (hasConstantIntegerDef(root.getOperand(0), 1)) {
      aliasedPayload = root.getOperation()->getOperand(1);
      aliasedValue = root.getOperand(1);
    } else if (hasConstantIntegerDef(root.getOperand(1), 1)) {
      aliasedPayload = root.getOperation()->getOperand(0);
      aliasedValue = root.getOperand(0);
    } else {
      return mlir::failure();
    }

    if (root.getResult(0).isEquivalentTo(aliasedValue))
      return mlir::failure();

    ++matchCount;
    return rewriter.replaceOp(root, {aliasedPayload});
  }

private:
  unsigned &matchCount;
};

mlir::LogicalResult importFunctionToEGraph(mlir::ModuleOp module) {
  if (!module.lookupSymbol<mlir::func::FuncOp>("arith_demo"))
    return module.emitError("expected a func.func named @arith_demo");

  mlir::PassManager pm(module.getContext());
  pm.addPass(mlir::egraph::createConvertFuncToEGraphPass());
  return pm.run(module.getOperation());
}

mlir::LogicalResult runDemo(mlir::ModuleOp module) {
  if (mlir::failed(importFunctionToEGraph(module)))
    return module.emitError("failed to import @arith_demo into egraph");

  auto originalFunc = module.lookupSymbol<mlir::func::FuncOp>("arith_demo");
  if (!originalFunc)
    return module.emitError("expected @arith_demo after import");
  mlir::Block &originalBlock = originalFunc.getBody().front();

  auto egraph =
      module.lookupSymbol<mlir::egraph::EGraphOp>("arith_demo_egraph");
  if (!egraph)
    return module.emitError("expected @arith_demo_egraph after import");

  mlir::egraph::EGraph graph;
  if (mlir::failed(graph.indexEGraph(egraph)))
    return egraph.emitOpError("failed to index egraph");

  unsigned mulByTwoMatches = 0;
  unsigned reassociateDivMatches = 0;
  unsigned divSelfMatches = 0;
  unsigned mulByOneMatches = 0;

  mlir::egraph::EGraphPatternSet patterns;
  patterns.add<MulByTwoToShiftPattern>(mulByTwoMatches);
  patterns.add<ReassociateDivPattern>(reassociateDivMatches);
  patterns.add<DivSelfToOnePattern>(divSelfMatches);
  patterns.add<MulByOneToAliasPattern>(mulByOneMatches);

  mlir::FailureOr<mlir::egraph::EGraphRewriteDriverResult> result =
      mlir::egraph::applyEGraphPatterns(graph, patterns, module);
  if (mlir::failed(result))
    return module.emitError("failed to apply egraph patterns");

  mlir::FailureOr<mlir::egraph::EGraphExtractResult> selection =
      extractDemoGraph(graph, egraph);
  if (mlir::failed(selection))
    return egraph.emitOpError("failed to extract graph");

  if (mlir::failed(rewriteBlockWithExtractedGraph(originalBlock, graph, egraph,
                                                  *selection)))
    return mlir::failure();

  egraph.erase();

  llvm::outs() << "Pattern matches:\n";
  llvm::outs() << "  muli(x, 2) -> shli(x, 1): " << mulByTwoMatches << "\n";
  llvm::outs() << "  div(mul(x, y), z) -> mul(x, div(y, z)): "
               << reassociateDivMatches << "\n";
  llvm::outs() << "  div(x, x) -> 1: " << divSelfMatches << "\n";
  llvm::outs() << "  mul(x, 1) -> x: " << mulByOneMatches << "\n";

  llvm::outs() << "Driver stats: ";
  mlir::egraph::printEGraphRewriteDriverResult(llvm::outs(), *result);
  llvm::outs() << "\nOptimized IR:\n";
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
