#include "DialectWrapper.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace cicero_compiler::dialect;

#include "Dialect.cpp.inc"

void CiceroDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Ops.cpp.inc"
