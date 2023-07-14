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

## Dialect Design

```mlir

// Example: ab|cd

module {
    "cicero.regionSplit"() ({
        // Match a
        "cicero.match_char"() {targetChar = 97 : i8} : () -> ()
        // Match b
        "cicero.match_char"() {targetChar = 98 : i8} : () -> ()
    }) { splitReturn = "SPLIT0_RETURN"} : () -> ()

    // Match c
    "cicero.match_char"() {targetChar = 99 : i8} : () -> ()
    // Match d
    "cicero.match_char"() {targetChar = 100 : i8} : () -> ()

    "cicero.placeholder"() {sym_name = "SPLIT0_RETURN"} : () -> ()
    "cicero.accept"() : () -> ()

    "cicero.placeholder"() {sym_name = "CICERO_END"} : () -> ()
}

// Pass to flatten the `regionSplit` into a `targetSplit`

module {
    // Splits execution between next instruction (match c) and the one
    // with "SPLIT0_TARGET" symbol
    "cicero.targetSplit"() {splitTarget = "SPLIT0_TARGET"} : () -> ()

    // Match c
    "cicero.match_char"() {targetChar = 99 : i8} : () -> ()
    // Match d
    "cicero.match_char"() {targetChar = 100 : i8} : () -> ()

    "cicero.placeholder"() {sym_name = "SPLIT0_TARGET"} : () -> ()
    "cicero.accept"() {} : () -> ()
    "cicero.placeholder"() {sym_name = "CICERO_END"} : () -> ()

    //// This is the body of SPLIT0
    // Match a
    "cicero.match_char"() {sym_name = "SPLIT0_TARGET", targetChar = 97 : i8} : () -> ()
    // Match b
    "cicero.match_char"() {targetChar = 98 : i8} : () -> ()
    // Return
    "cicero.jump"() {target = "SPLIT0_RETURN"} : () -> ()
    /// End body of SPLIT0

    "cicero.placeholder"() {sym_name = "CICERO_END"} : () -> ()
}
```

## new dialect design


```mlir

// Example: (ab|cd)*

"builtin.module"() ({
  "cicero.split"() ({
    "cicero.split"() ({
      "cicero.match_char"() {targetChar = 97 : i8} : () -> ()
      "cicero.match_char"() {targetChar = 98 : i8} : () -> ()
      "cicero.jump"() {target = @S1} : () -> ()
    }) : () -> ()
    "cicero.match_char"() {targetChar = 99 : i8} : () -> ()
    "cicero.match_char"() {targetChar = 100 : i8} : () -> ()
    "cicero.placeholder"() {sym_name = "S1"} : () -> ()
    "cicero.jump"() {target = @S0} : () -> ()
  }) {sym_name = "S0"} : () -> ()
  "cicero.accept"() : () -> ()
}) : () -> ()

// Flatten the (inner) split

"builtin.module"() ({
  "cicero.split"() ({
    "cicero.flatsplit"() {splitTarget = "S2"} : () -> ()
    "cicero.match_char"() {targetChar = 99 : i8} : () -> ()
    "cicero.match_char"() {targetChar = 100 : i8} : () -> ()
    "cicero.placeholder"() {sym_name = "S1"} : () -> ()
    "cicero.jump"() {target = @S0} : () -> ()
    // Body of inner split
    "cicero.placeholder"() {sym_name = "S2"} : () -> ()
    "cicero.match_char"() {targetChar = 97 : i8} : () -> ()
    "cicero.match_char"() {targetChar = 98 : i8} : () -> ()
    "cicero.jump"() {target = @S1} : () -> ()
    // End body of inner split
  }) {sym_name = "S0"} : () -> ()
  "cicero.accept"() : () -> ()
}) : () -> ()

// Flatten the outer split

"builtin.module"() ({
  "cicero.flatsplit"() {splitTarget = "S3"} : () -> ()
  "cicero.accept"() : () -> ()
  // Body of outer split
  "cicero.placeholder"() {sym_name = "S3"} : () -> ()
  "cicero.flatsplit"() {splitTarget = "S2"} : () -> ()
  "cicero.match_char"() {targetChar = 99 : i8} : () -> ()
  "cicero.match_char"() {targetChar = 100 : i8} : () -> ()
  "cicero.placeholder"() {sym_name = "S1"} : () -> ()
  "cicero.jump"() {target = @S0} : () -> ()
  // Body of inner split
  "cicero.placeholder"() {sym_name = "S2"} : () -> ()
  "cicero.match_char"() {targetChar = 97 : i8} : () -> ()
  "cicero.match_char"() {targetChar = 98 : i8} : () -> ()
  "cicero.jump"() {target = @S1} : () -> ()
  // End body of inner split
  // End body of outer split
}) : () -> ()

```