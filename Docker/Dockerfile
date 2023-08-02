FROM ubuntu:22.04

run mkdir -p /etc/apt/sources.list.d

run apt-get update && apt-get install -y ca-certificates gnupg
run apt-key adv --recv-key --keyserver keyserver.ubuntu.com 15CF4D18AF4F7421
run echo 'deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main' > /etc/apt/sources.list.d/llvm.list
run apt-get update && apt-get install -y libmlir-16-dev mlir-16-tools llvm-16-dev antlr4 libantlr4-runtime-dev cmake clang-16