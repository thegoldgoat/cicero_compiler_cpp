# Cicero Compiler CPP

C++ implementation of a regex compiler for [Cicero](https://github.com/necst/cicero).

## Build

Install dependencies first, on Fedora Linux:

```bash
dnf install cmake antlr4 antlr4-cpp-runtime-devel mlir-devel
```

1. `cmake`: cross platform build file generator
2. `antlr4`: tool for building parser/lexer from declarative grammar/tokens
3. `antlr4-cpp-runtime-devel`: C++ runtime for antlr4
4. `mlir-devel`: intermediate representation library

Once dependencies are installed, clone this repo and `cd` into it:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```