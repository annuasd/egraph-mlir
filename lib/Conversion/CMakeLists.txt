add_mlir_library(EGraphConversion
  FuncToEGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_EGRAPH_INCLUDE_DIR}/MLIREGraph

  LINK_LIBS PUBLIC
  EGraphIR
  EGraph
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
)
