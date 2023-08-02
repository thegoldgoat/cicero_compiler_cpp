set -e

cd app

mkdir -p build
cd build

cmake .. -DCMAKE_CXX_COMPILER=$(which clang++-16) -DMLIR_DIR=/usr/lib/llvm-16/lib/cmake/mlir

# Actual build
make -j $(nproc)

# Tests
make test