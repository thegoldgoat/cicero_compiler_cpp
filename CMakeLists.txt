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
include_directories(${CMAKE_CURRENT_BINARY_DIR}/Parser/src/mlir-dialect)

include_directories(include)
include_directories(Parser/include)

add_executable(
  ciceroc
  src/ciceroc.cpp
  src/mlir-dialect/CiceroDialectWrapper.cpp
  src/CiceroMLIRGenerator.cpp
  src/CiceroPasses.cpp
  src/cicero_helper.cpp
)

add_dependencies(
  ciceroc
  CiceroDialect
  RegexDialect
)

link_directories(${LLVM_BUILD_LIBRARY_DIR})

target_link_libraries(ciceroc
  PRIVATE
    MLIRRegexParser
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)

add_executable(
  objdump
  src/tool/objdump.cpp
)

# Tests

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(thirdparty/SoftwareCicero)
    add_subdirectory(tests)
endif()