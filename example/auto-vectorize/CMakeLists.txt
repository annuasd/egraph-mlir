set(LLVM_LINK_COMPONENTS
  Support
)

add_mlir_tool(auto-vectorize-example
  auto-vectorize.cpp

  DEPENDS
  EGraph
  EGraphIR
)

target_compile_definitions(auto-vectorize-example PRIVATE
  AUTO_VECTORIZE_EXAMPLE_INPUT="${CMAKE_CURRENT_SOURCE_DIR}/auto-vectorize.mlir"
)

target_link_libraries(auto-vectorize-example PRIVATE
  EGraph
  EGraphIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRParser
  MLIRPass
  MLIRVectorDialect
)
llvm_update_compile_flags(auto-vectorize-example)
mlir_check_all_link_libraries(auto-vectorize-example)
