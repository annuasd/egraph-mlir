add_mlir_dialect_library(EGraphTestDialect
  EGraphTestDialect.cpp
  EGraphTestOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_EGRAPH_TEST_INCLUDE_DIR}/EGraphTest

  DEPENDS
  EGraphTestOpsIncGen

  LINK_LIBS PUBLIC
  MLIRBytecodeOpInterface
  MLIRIR
  MLIRSideEffectInterfaces
)

target_include_directories(EGraphTestDialect
  PUBLIC
  ${MLIR_EGRAPH_TEST_INCLUDE_DIR}
  ${MLIR_EGRAPH_TEST_GENERATED_INCLUDE_DIR})
