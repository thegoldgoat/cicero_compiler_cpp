cmake_minimum_required(VERSION 3.20)

project("CiceroCompilerCpp")

set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${MLIR_INCLUDE_DIRS})

add_subdirectory(Parser)
add_subdirectory(src/mlir-dialect)

# Include generated dialect headers
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src/mlir-dialect)
include_directories(include)
include_directories(Parser/include)

add_executable(
  main
  ${CMAKE_CURRENT_BINARY_DIR}/src/mlir-dialect/Dialect.cpp.inc
  src/main.cpp
  src/mlir-dialect/DialectWrapper.cpp
  src/MLIRGenerator.cpp
)

add_dependencies(
  main
  CiceroDialect
)

link_directories(${LLVM_BUILD_LIBRARY_DIR})

target_link_libraries(main
  PRIVATE
    RegexParser
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)