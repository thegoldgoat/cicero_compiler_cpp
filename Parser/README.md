# Regex Parser

This directory contains two libraries that parse a regular expression and output different representations.

## ASTRegexParser

The `ASTRegexParser` library parses a regular expression and outputs a custom Abstract Syntax Tree (AST) representation.

To use the `ASTRegexParser` library, include the `ASTParser.h` header file and link against the `ASTRegexParser` library.

## MLIRRegexParser

The `MLIRRegexParser` library parses a regular expression and outputs an MLIR representation using a custom dialect. MLIR is a multi-level intermediate representation that can be used to represent and transform code in a variety of languages and domains.

To use the `MLIRRegexParser` library, include the `MLIRParser.h` header file and link against the `MLIRRegexParser` library.

## Dependencies

To build the libraries, you will need the following dependencies:

1. cmake
2. mlir and llvm
3. antlr4 (both tools and C++ runtime)

## Building

To build the libraries, run the following commands:

```bash

mkdir build
cd build
cmake ..
cmake --build .
```
