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

### Build with Docker

If you prefer, you can use a Docker image to build. You can find in `Docker/Dockerfile` the dockerfile to build the image:

```bash
# Build docker image which contains build dependencies
docker build -t cicero_build_environment:latest Docker

# Run build and test within docker image
docker run -v $PWD:/app cicero_build_environment:latest /bin/bash /app/Docker/build_and_test.sh
```

## Dialect Design

Here is an example before and after the canonicalization of the `cicero.split` operation, which
gets replaced by a `cicero.flat_split` operation. The `cicero.flat_split` operation has a reference
to the actual body of the split, which originally was the body of the `cicero.split` operation.

```mlir

// Example: (ab|cd)|ef
--- mlir before any pattern rewrites ---

module {
  cicero.split {splitReturn = @PREFIX_SPLIT, sym_name = "PREFIX_SPLIT"} {
    cicero.match_any
  }
  cicero.split {splitReturn = @S0} {
    cicero.split {splitReturn = @S1} {
      cicero.match_char a
      cicero.match_char b
    }
    cicero.match_char c
    cicero.match_char d
    cicero.placeholder {sym_name = "S1"}
  }
  cicero.match_char e
  cicero.match_char f
  cicero.placeholder {sym_name = "S0"}
  cicero.accept_partial
}

--- mlir after pattern rewrites      ---

module {
  cicero.flat_split {splitTarget = @F2, sym_name = "PREFIX_SPLIT"}
  cicero.flat_split {splitTarget = @F0}
  cicero.match_char e
  cicero.match_char f
  cicero.accept_partial {sym_name = "S0"}
  cicero.flat_split {splitTarget = @F1, sym_name = "F0"}
  cicero.match_char c
  cicero.match_char d
  cicero.jump {sym_name = "S1", target = @S0}
  cicero.match_char {sym_name = "F1"} a
  cicero.match_char b
  cicero.jump {target = @S1}
  cicero.match_any {sym_name = "F2"}
  cicero.jump {target = @PREFIX_SPLIT}
}

```
