set(LLVM_LINK_COMPONENTS
  Support
)

add_mlir_library(EGraphTestPasses
  TestPasses.cpp

  LINK_LIBS PUBLIC
  EGraphConversion
  EGraph
  EGraphIR
  EGraphTestDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
