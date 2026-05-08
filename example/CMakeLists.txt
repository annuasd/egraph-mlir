add_subdirectory(arith)
add_subdirectory(transpose)

add_custom_target(mlir-egraph-example
  DEPENDS
  arith-example
  transpose-example
)
