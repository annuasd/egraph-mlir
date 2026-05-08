set(LLVM_LINK_COMPONENTS
  Support
)

add_mlir_tool(transpose-demo
  transpose.cpp

  DEPENDS
  EGraph
  EGraphIR
)

target_compile_definitions(transpose-demo PRIVATE
  TRANSPOSE_DEMO_INPUT="${CMAKE_CURRENT_SOURCE_DIR}/transpose.mlir"
)

target_link_libraries(transpose-demo PRIVATE
  EGraph
  EGraphIR
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTosaDialect
)
llvm_update_compile_flags(transpose-demo)
mlir_check_all_link_libraries(transpose-demo)
