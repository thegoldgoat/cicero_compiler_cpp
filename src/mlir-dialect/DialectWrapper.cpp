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

#include <iostream>
LogicalResult JumpOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if(nullptr != symbolTable.lookupNearestSymbolFrom(getOperation(), getTargetAttrName())) {
    return success();
  } else {
    return failure();
  }
}

#define GET_OP_CLASSES
#include "Ops.cpp.inc"
