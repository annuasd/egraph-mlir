add_mlir_dialect_library(EGraphIR
  EGraphDialect.cpp
  EGraphOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_EGRAPH_INCLUDE_DIR}/MLIREGraph

  DEPENDS
  EGraphOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
