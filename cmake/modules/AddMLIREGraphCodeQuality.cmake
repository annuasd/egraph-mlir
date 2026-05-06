# Add project-local formatting and static-analysis targets.

function(mlir_egraph_collect_code_quality_files out_format_files out_tidy_files)
  file(GLOB_RECURSE format_files CONFIGURE_DEPENDS
    "${MLIR_EGRAPH_SOURCE_DIR}/include/*.h"
    "${MLIR_EGRAPH_SOURCE_DIR}/include/*.hpp"
    "${MLIR_EGRAPH_SOURCE_DIR}/include/*.inc"
    "${MLIR_EGRAPH_SOURCE_DIR}/include/*.td"
    "${MLIR_EGRAPH_SOURCE_DIR}/lib/*.cpp"
    "${MLIR_EGRAPH_SOURCE_DIR}/lib/*.h"
    "${MLIR_EGRAPH_SOURCE_DIR}/lib/*.hpp"
    "${MLIR_EGRAPH_SOURCE_DIR}/lib/*.inc"
    "${MLIR_EGRAPH_SOURCE_DIR}/lib/*.td"
    "${MLIR_EGRAPH_SOURCE_DIR}/tools/*.cpp"
    "${MLIR_EGRAPH_SOURCE_DIR}/tools/*.h"
    "${MLIR_EGRAPH_SOURCE_DIR}/tools/*.hpp"
    "${MLIR_EGRAPH_SOURCE_DIR}/tools/*.inc"
    "${MLIR_EGRAPH_SOURCE_DIR}/tools/*.td"
    "${MLIR_EGRAPH_SOURCE_DIR}/test/*.cpp"
    "${MLIR_EGRAPH_SOURCE_DIR}/test/*.h"
    "${MLIR_EGRAPH_SOURCE_DIR}/test/*.hpp"
    "${MLIR_EGRAPH_SOURCE_DIR}/test/*.inc"
    "${MLIR_EGRAPH_SOURCE_DIR}/test/*.td")
  file(GLOB_RECURSE tidy_files CONFIGURE_DEPENDS
    "${MLIR_EGRAPH_SOURCE_DIR}/lib/*.cpp"
    "${MLIR_EGRAPH_SOURCE_DIR}/tools/*.cpp"
    "${MLIR_EGRAPH_SOURCE_DIR}/test/*.cpp")

  list(SORT format_files)
  list(SORT tidy_files)

  set(${out_format_files} ${format_files} PARENT_SCOPE)
  set(${out_tidy_files} ${tidy_files} PARENT_SCOPE)
endfunction()

function(mlir_egraph_add_missing_tool_target target_name tool_name cache_name)
  add_custom_target(${target_name}
    COMMAND ${CMAKE_COMMAND} -E echo
            "${tool_name} was not found. Reconfigure with -D${cache_name}=/path/to/${tool_name}."
    COMMAND ${CMAKE_COMMAND} -E false
    COMMENT "${target_name} requires ${tool_name}"
    VERBATIM)
endfunction()

function(mlir_egraph_add_code_quality_targets)
  mlir_egraph_collect_code_quality_files(format_files tidy_files)

  find_program(MLIR_EGRAPH_CLANG_FORMAT
    NAMES clang-format
    HINTS "${LLVM_TOOLS_BINARY_DIR}"
    DOC "Path to clang-format used by MLIR-EGraph quality targets")
  find_program(MLIR_EGRAPH_CLANG_TIDY
    NAMES clang-tidy
    HINTS "${LLVM_TOOLS_BINARY_DIR}"
    DOC "Path to clang-tidy used by MLIR-EGraph quality targets")

  if(format_files AND MLIR_EGRAPH_CLANG_FORMAT)
    add_custom_target(mlir-egraph-format
      COMMAND "${MLIR_EGRAPH_CLANG_FORMAT}" -i ${format_files}
      COMMENT "Formatting MLIR-EGraph C++ and TableGen files"
      VERBATIM)
    add_custom_target(check-mlir-egraph-format
      COMMAND "${MLIR_EGRAPH_CLANG_FORMAT}" --dry-run --Werror ${format_files}
      COMMENT "Checking MLIR-EGraph C++ and TableGen formatting"
      VERBATIM)
  elseif(format_files)
    mlir_egraph_add_missing_tool_target(mlir-egraph-format clang-format
                                        MLIR_EGRAPH_CLANG_FORMAT)
    mlir_egraph_add_missing_tool_target(check-mlir-egraph-format clang-format
                                        MLIR_EGRAPH_CLANG_FORMAT)
  else()
    add_custom_target(mlir-egraph-format
      COMMAND ${CMAKE_COMMAND} -E echo "No C++ or TableGen files to format."
      VERBATIM)
    add_custom_target(check-mlir-egraph-format
      COMMAND ${CMAKE_COMMAND} -E echo "No C++ or TableGen files to format."
      VERBATIM)
  endif()

  if(tidy_files AND MLIR_EGRAPH_CLANG_TIDY)
    add_custom_target(check-mlir-egraph-tidy
      COMMAND "${MLIR_EGRAPH_CLANG_TIDY}" -p "${CMAKE_BINARY_DIR}" ${tidy_files}
      COMMENT "Running clang-tidy on MLIR-EGraph C++ files"
      VERBATIM)
  elseif(tidy_files)
    mlir_egraph_add_missing_tool_target(check-mlir-egraph-tidy clang-tidy
                                        MLIR_EGRAPH_CLANG_TIDY)
  else()
    add_custom_target(check-mlir-egraph-tidy
      COMMAND ${CMAKE_COMMAND} -E echo "No C++ files to check with clang-tidy."
      VERBATIM)
  endif()

  add_custom_target(check-mlir-egraph-quality
    DEPENDS check-mlir-egraph-format check-mlir-egraph-tidy)
endfunction()
