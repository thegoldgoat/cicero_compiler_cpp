# Cicero Compiler CPP

C++ implementation of a regex compiler for [Cicero](https://github.com/necst/cicero).

## Build

Install dependencies first:

```bash
# Ubuntu Linux
# Add LLVM apt repository, follow instruction on https://apt.llvm.org/
apt install libmlir-16-dev mlir-16-tools llvm-16-dev antlr4 libantlr4-runtime-dev cmake

# Fedora Linux
dnf install cmake antlr4 antlr4-cpp-runtime-devel mlir-devel llvm-devel
```

1. `cmake`: cross platform build file generator
2. `antlr4`: tool for building parser/lexer from declarative grammar/tokens
3. `antlr4-cpp-runtime-devel`: C++ runtime for antlr4
4. `mlir-devel`: intermediate representation library
5. `llvm-devel`: compiler infrastructure library

Once dependencies are installed, clone this repo and `cd` into it:

```bash
mkdir build
cd build
# Optional, only if you want to build tests
git submodule update --init --recursive
# If you don't want to build tests, add `-DBUILD_TESTING=OFF` to the next command
cmake ..
cmake --build .
```

## Dialect Design

Here is an example before and after the canonicalization of the `cicero.split` operation, which
gets replaced by a `cicero.flat_split` operation. The `cicero.flat_split` operation has a reference 
to the actual body of the split, which originally was the body of the `cicero.split` operation.

```mlir

// Example: (ab|cd)|ef

// === Module before canonicalization ===

module {
  "cicero.split"() ({
    "cicero.split"() ({
      // Match a
      "cicero.match_char"() {targetChar = 97 : i8} : () -> ()
      // Match b
      "cicero.match_char"() {targetChar = 98 : i8} : () -> ()
      "cicero.jump"() {target = @S1} : () -> ()
    }) {splitReturn = @S1} : () -> ()
    // Match c
    "cicero.match_char"() {targetChar = 99 : i8} : () -> ()
    // Match d
    "cicero.match_char"() {targetChar = 100 : i8} : () -> ()
    "cicero.placeholder"() {sym_name = "S1"} : () -> ()
    "cicero.jump"() {target = @S0} : () -> ()
  }) {splitReturn = @S0} : () -> ()
  // Match e
  "cicero.match_char"() {targetChar = 101 : i8} : () -> ()
  // Match f
  "cicero.match_char"() {targetChar = 102 : i8} : () -> ()
  "cicero.placeholder"() {sym_name = "S0"} : () -> ()
  "cicero.accept"() : () -> ()
}

// === Module after canonicalization ===

module {
  "cicero.flat_split"() {splitTarget = @FLATTEN_0} : () -> ()
  // Match e
  "cicero.match_char"() {targetChar = 101 : i8} : () -> ()
  // Match f
  "cicero.match_char"() {targetChar = 102 : i8} : () -> ()
  "cicero.placeholder"() {sym_name = "S0"} : () -> ()
  "cicero.accept"() : () -> ()
  "cicero.placeholder"() {sym_name = "FLATTEN_0"} : () -> ()
  "cicero.flat_split"() {splitTarget = @FLATTEN_1} : () -> ()
  // Match c
  "cicero.match_char"() {targetChar = 99 : i8} : () -> ()
  // Match d
  "cicero.match_char"() {targetChar = 100 : i8} : () -> ()
  "cicero.placeholder"() {sym_name = "S1"} : () -> ()
  "cicero.jump"() {target = @S0} : () -> ()
  "cicero.placeholder"() {sym_name = "FLATTEN_1"} : () -> ()
  // Match a
  "cicero.match_char"() {targetChar = 97 : i8} : () -> ()
  // Match b
  "cicero.match_char"() {targetChar = 98 : i8} : () -> ()
  "cicero.jump"() {target = @S1} : () -> ()
}

```