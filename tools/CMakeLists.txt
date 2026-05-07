set(LLVM_LINK_COMPONENTS
  Support
)

set(LIBS
  EGraphIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIROptLib
  MLIRPass
)

add_mlir_tool(mlir-egraph-opt
  mlir-egraph-opt.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  ${LIBS}
)
target_link_libraries(mlir-egraph-opt PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-egraph-opt)
mlir_check_all_link_libraries(mlir-egraph-opt)
