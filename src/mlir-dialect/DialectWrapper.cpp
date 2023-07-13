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

LogicalResult myLookup(SymbolTableCollection &symbolTable, Operation *op,
                       FlatSymbolRefAttr &symbol) {
    while (op && !op->hasTrait<OpTrait::SymbolTable>()) {
        op = op->getParentOp();
    }

    if (!op) {
        return failure();
    }

    if (symbolTable.lookupSymbolIn(op, symbol)) {
        return success();
    } else {
        return myLookup(symbolTable, op->getParentOp(), symbol);
    }
}

LogicalResult JumpOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    auto symbolName = getTargetAttr().getValue().str();

    auto symbol = getTargetAttr();
    return myLookup(symbolTable, getOperation(), symbol);
}

#define GET_OP_CLASSES
#include "Ops.cpp.inc"
