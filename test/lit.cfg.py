# -*- Python -*-

import os

import lit.formats

from lit.llvm import llvm_config

config.name = "EGraph"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.mlir_egraph_obj_root, "test")

config.excludes = [
    "Inputs",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "include",
    "lib",
]

def cmake_bool(value):
    return str(value).upper() in {"1", "ON", "TRUE", "YES", "Y"}


if cmake_bool(getattr(config, "mlir_egraph_enable_z3", "OFF")):
    config.available_features.add("z3")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

tool_dirs = [config.mlir_egraph_tools_dir, config.llvm_tools_dir]
tools = [
    "FileCheck",
    "count",
    "mlir-egraph-opt",
    "transpose-example",
    "not",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
