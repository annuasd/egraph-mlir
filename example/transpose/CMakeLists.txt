set(LLVM_LINK_COMPONENTS
  Support
)

add_mlir_tool(transpose-example
  transpose.cpp

  DEPENDS
  EGraph
  EGraphIR
)

target_compile_definitions(transpose-example PRIVATE
  TRANSPOSE_EXAMPLE_INPUT="${CMAKE_CURRENT_SOURCE_DIR}/transpose.mlir"
)

target_link_libraries(transpose-example PRIVATE
  EGraph
  EGraphIR
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTosaDialect
)
llvm_update_compile_flags(transpose-example)
mlir_check_all_link_libraries(transpose-example)
