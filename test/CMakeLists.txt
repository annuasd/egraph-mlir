set(MLIR_EGRAPH_TOOLS_DIR "${LLVM_RUNTIME_OUTPUT_INTDIR}")
set(MLIR_EGRAPH_TEST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(MLIR_EGRAPH_TEST_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")

include_directories(${MLIR_EGRAPH_TEST_INCLUDE_DIR})
include_directories(${MLIR_EGRAPH_TEST_GENERATED_INCLUDE_DIR})

add_subdirectory(include)
add_subdirectory(lib)

target_compile_definitions(mlir-egraph-opt PRIVATE MLIR_EGRAPH_ENABLE_TEST_DIALECT)
target_link_libraries(mlir-egraph-opt PRIVATE
  MLIRControlFlowDialect
  EGraphTestDialect
  EGraphTestPasses)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
file(MAKE_DIRECTORY
  ${CMAKE_CURRENT_BINARY_DIR}/API
  ${CMAKE_CURRENT_BINARY_DIR}/Driver
  ${CMAKE_CURRENT_BINARY_DIR}/IR)

set(MLIR_EGRAPH_TEST_DEPENDS
  mlir-egraph-opt
  transpose-example)
set(MLIR_EGRAPH_LIT_SKIP
  "^API(/.*)?$"
  "^Driver(/.*)?$"
  "^IR(/.*)?$"
  "^include(/.*)?$"
  "^lib(/.*)?$")

add_lit_testsuite(check-mlir-egraph "Running the MLIR-EGraph regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_EGRAPH_TEST_DEPENDS})

add_lit_testsuite(check-mlir-egraph-api "Running the MLIR-EGraph API tests"
  ${CMAKE_CURRENT_BINARY_DIR}/API
  DEPENDS ${MLIR_EGRAPH_TEST_DEPENDS})

add_lit_testsuite(check-mlir-egraph-driver "Running the MLIR-EGraph driver tests"
  ${CMAKE_CURRENT_BINARY_DIR}/Driver
  DEPENDS ${MLIR_EGRAPH_TEST_DEPENDS})

add_lit_testsuite(check-mlir-egraph-ir "Running the MLIR-EGraph IR tests"
  ${CMAKE_CURRENT_BINARY_DIR}/IR
  DEPENDS ${MLIR_EGRAPH_TEST_DEPENDS})

add_lit_testsuites(MLIR-EGRAPH ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_EGRAPH_TEST_DEPENDS}
  SKIP ${MLIR_EGRAPH_LIT_SKIP})
