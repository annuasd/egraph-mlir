set(LLVM_LINK_COMPONENTS
  Support
)

add_mlir_tool(arith-demo
  arith.cpp

  DEPENDS
  EGraph
  EGraphIR
)

target_compile_definitions(arith-demo PRIVATE
  ARITH_DEMO_INPUT="${CMAKE_CURRENT_SOURCE_DIR}/arith.mlir"
)

target_link_libraries(arith-demo PRIVATE
  EGraph
  EGraphIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
)
llvm_update_compile_flags(arith-demo)
mlir_check_all_link_libraries(arith-demo)
