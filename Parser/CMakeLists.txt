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
        RegexParser
        SHARED
        ${PROJECT_BINARY_DIR}/antlr-grammar/regexLexer.cpp
        ${PROJECT_BINARY_DIR}/antlr-grammar/regexParser.cpp
        ${PROJECT_SOURCE_DIR}/src/ASTParser.cpp
        ${PROJECT_SOURCE_DIR}/src/visitor/Visitor.cpp
        ${PROJECT_BINARY_DIR}/include/metachars.h
)

target_link_libraries(
        RegexParser
        antlr4-runtime
)

add_executable(parsertest
        src/main.cpp
)

target_link_libraries(
        parsertest
        RegexParser
)
