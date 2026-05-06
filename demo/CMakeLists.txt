set(LLVM_LINK_COMPONENTS
  Support
)

add_mlir_tool(mlir-egraph-demo
  demo.cpp

  DEPENDS
  EGraph
  EGraphIR
)

target_compile_definitions(mlir-egraph-demo PRIVATE
  MLIR_EGRAPH_DEMO_INPUT="${CMAKE_CURRENT_SOURCE_DIR}/demo.mlir"
)

target_link_libraries(mlir-egraph-demo PRIVATE
  EGraph
  EGraphIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
)
llvm_update_compile_flags(mlir-egraph-demo)
mlir_check_all_link_libraries(mlir-egraph-demo)
