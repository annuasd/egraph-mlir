set(LLVM_LINK_COMPONENTS
  Support
)

add_mlir_tool(arith-example
  arith.cpp

  DEPENDS
  EGraph
  EGraphIR
)

target_compile_definitions(arith-example PRIVATE
  ARITH_EXAMPLE_INPUT="${CMAKE_CURRENT_SOURCE_DIR}/arith.mlir"
)

target_link_libraries(arith-example PRIVATE
  EGraph
  EGraphIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
)
llvm_update_compile_flags(arith-example)
mlir_check_all_link_libraries(arith-example)
