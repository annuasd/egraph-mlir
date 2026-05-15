cmake_minimum_required(VERSION 3.20.0)

project(mlir-egraph LANGUAGES C CXX)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile_commands.json")
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")
option(MLIR_EGRAPH_INCLUDE_TESTS "Generate MLIR-EGraph test targets" ON)

find_package(MLIR REQUIRED CONFIG)
option(MLIR_EGRAPH_ENABLE_Z3 "Enable the Z3 solver backend" OFF)
option(MLIR_EGRAPH_ENABLE_OR_TOOLS "Enable the OR-Tools solver backend" OFF)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

if(MLIR_EGRAPH_ENABLE_Z3)
  find_package(Z3 CONFIG REQUIRED)

  if(NOT TARGET z3::libz3)
    message(FATAL_ERROR
      "Z3 was found, but the imported target z3::libz3 is unavailable. "
      "Use a Z3 package that exports the CMake config target.")
  endif()

  message(STATUS "Using Z3 solver backend target: z3::libz3")
else()
  message(STATUS "Z3 solver backend disabled")
endif()

if(MLIR_EGRAPH_ENABLE_OR_TOOLS)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
  find_package(ortools CONFIG REQUIRED)
else()
  message(STATUS "OR-Tools solver backend disabled")
endif()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(MLIR_EGRAPH_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(MLIR_EGRAPH_BINARY_DIR ${PROJECT_BINARY_DIR})
set(MLIR_EGRAPH_INCLUDE_DIR ${MLIR_EGRAPH_SOURCE_DIR}/include)
set(MLIR_EGRAPH_GENERATED_INCLUDE_DIR ${MLIR_EGRAPH_BINARY_DIR}/include)

list(APPEND CMAKE_MODULE_PATH "${MLIR_EGRAPH_SOURCE_DIR}/cmake/modules")
include(AddMLIREGraphCodeQuality)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(example)
if(MLIR_EGRAPH_INCLUDE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

mlir_egraph_add_code_quality_targets()
