cmake_minimum_required(VERSION 3.7)

project("RegexParser")

# compiler must be at least 17
set(CMAKE_CXX_STANDARD 17)

# add ANTLR Executable path
set(ANTLR_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/tools/antlr-4.10.1-complete.jar)

# Include generated antlr4 files
include_directories(src/antlrlib)
# Include antlr4 runtime
include_directories(/usr/include/antlr4-runtime)

include_directories(include)

add_library(
        RegexParser
        SHARED
        src/antlrlib/regexLexer.cpp
        src/antlrlib/regexParser.cpp
)

target_link_libraries(
        RegexParser
        -lantlr4-runtime
)

add_executable(parsertest
        src/main.cpp
        src/visitor/Visitor.cpp
)

target_link_libraries(
        parsertest
        RegexParser
)
