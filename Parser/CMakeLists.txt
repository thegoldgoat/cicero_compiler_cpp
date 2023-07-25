cmake_minimum_required(VERSION 3.20)

project("RegexParser")

# compiler must be at least 17
set(CMAKE_CXX_STANDARD 17)

# Include antlr4 runtime
find_path(
    ANTLR4_RUNTIME_INCLUDE_DIR
    antlr4-runtime/antlr4-runtime.h
    REQUIRED
)

message(STATUS "Using ANTLR4_RUNTIME_INCLUDE_DIR: ${ANTLR4_RUNTIME_INCLUDE_DIR}")

include_directories("${ANTLR4_RUNTIME_INCLUDE_DIR}/antlr4-runtime")

find_program(
    ANTLR4_TOOL
    antlr4
    REQUIRED
)

find_library(
    ANTLR4_RUNTIME
    antlr4-runtime
    REQUIRED
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
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
endif()

add_subdirectory(src/mlir-dialect)

message("Include mlir-dialect ${CMAKE_CURRENT_BINARY_DIR}/src/mlir-dialect")
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src/mlir-dialect)

include_directories(include)

add_custom_command(
    OUTPUT
        ${PROJECT_BINARY_DIR}/antlr-grammar/regexLexer.cpp
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/antlr-grammar/regexLexer.g4
    COMMAND
        ${ANTLR4_TOOL}
        -Dlanguage=Cpp
        -no-listener
        -visitor
        -o ${PROJECT_BINARY_DIR}/antlr-grammar
        ${CMAKE_CURRENT_SOURCE_DIR}/src/antlr-grammar/regexLexer.g4
)

add_custom_command(
    OUTPUT
        ${PROJECT_BINARY_DIR}/antlr-grammar/regexParser.cpp
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/antlr-grammar/regexParser.g4
        ${PROJECT_BINARY_DIR}/antlr-grammar/regexLexer.cpp
    COMMAND
        ${ANTLR4_TOOL}
        -Dlanguage=Cpp
        -no-listener
        -visitor
        -o ${PROJECT_BINARY_DIR}/antlr-grammar
        -lib ${PROJECT_BINARY_DIR}/antlr-grammar
        ${CMAKE_CURRENT_SOURCE_DIR}/src/antlr-grammar/regexParser.g4
)

include_directories(${PROJECT_BINARY_DIR}/antlr-grammar)

add_custom_command(
    OUTPUT
        ${PROJECT_BINARY_DIR}/include/metachars.h
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_metachar_range.py
    COMMAND
        python3
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_metachar_range.py
        ${PROJECT_BINARY_DIR}/include/metachars.h
)

include_directories(${PROJECT_BINARY_DIR}/include)

add_library(
        MLIRRegexParser
        SHARED
        ${PROJECT_BINARY_DIR}/antlr-grammar/regexLexer.cpp
        ${PROJECT_BINARY_DIR}/antlr-grammar/regexParser.cpp
        ${PROJECT_SOURCE_DIR}/src/MLIRParser.cpp
        ${PROJECT_SOURCE_DIR}/src/RegexDialectWrapper.cpp
        ${PROJECT_BINARY_DIR}/include/metachars.h
        src/visitor/MLIRVisitor.cpp
        src/MLIRRegexOptimization.cpp
        src/passes/SubRegexPasses.cpp
)

add_dependencies(
    MLIRRegexParser
    RegexDialect
)

target_link_libraries(
        MLIRRegexParser
        antlr4-runtime
        MLIRAnalysis
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRTransforms
)

add_executable(
        mlirdumper
        src/mlir_dump.cpp
)

target_link_libraries(
        mlirdumper
        MLIRRegexParser
)