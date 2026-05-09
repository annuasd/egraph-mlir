add_subdirectory(arith)
add_subdirectory(auto-vectorize)
add_subdirectory(transpose)

add_custom_target(mlir-egraph-example
  DEPENDS
  arith-example
  auto-vectorize-example
  transpose-example
)
