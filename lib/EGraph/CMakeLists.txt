add_mlir_library(EGraph
  EGraph.cpp
  Extract.cpp
  Pattern.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_EGRAPH_INCLUDE_DIR}/MLIREGraph

  LINK_LIBS PUBLIC
  EGraphIR
  MLIRIR
)

if(MLIR_EGRAPH_ENABLE_Z3)
  target_compile_definitions(obj.EGraph PRIVATE MLIR_EGRAPH_ENABLE_Z3)
  target_link_libraries(obj.EGraph PUBLIC z3::libz3)

  target_compile_definitions(EGraph PUBLIC MLIR_EGRAPH_ENABLE_Z3)
  target_link_libraries(EGraph PUBLIC z3::libz3)
endif()
