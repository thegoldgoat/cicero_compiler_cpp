#include "CiceroDialectWrapper.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace cicero_compiler::dialect;

#include "CiceroDialect.cpp.inc"

void CiceroDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "CiceroOps.cpp.inc"
        >();
}

#include "MatchCharHelper.h"

DEFINE_MATCH_CHAR_PARSER_MACRO(MatchCharOp)
DEFINE_MATCH_CHAR_PRINTER_MACRO(MatchCharOp)

DEFINE_MATCH_CHAR_PARSER_MACRO(NotMatchCharOp)
DEFINE_MATCH_CHAR_PRINTER_MACRO(NotMatchCharOp)

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
    auto symbol = getTargetAttr();
    auto verifyResult = myLookup(symbolTable, getOperation(), symbol);
    if (failed(verifyResult)) {
        getOperation()->emitError(
            "Verification of JumpOp failed: jump target symbol = \"@" +
            symbol.getValue().str() + "\" not found.");
    }

    return verifyResult;
}

LogicalResult SplitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    auto symbol = getSplitReturnAttr();
    auto verifyResult = myLookup(symbolTable, getOperation(), symbol);
    if (failed(verifyResult)) {
        getOperation()->emitError(
            "Verification of SplitOp failed: return symbol = \"@" +
            symbol.getValue().str() + "\" not found.");
    }

    return verifyResult;
}

LogicalResult FlatSplitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    auto symbol = getSplitTargetAttr();
    auto verifyResult = myLookup(symbolTable, getOperation(), symbol);
    if (failed(verifyResult)) {
        getOperation()->emitError(
            "Verification of FlatSplitOp failed: split target symbol = \"@" +
            symbol.getValue().str() + "\" not found.");
    }

    return verifyResult;
}

#define GET_OP_CLASSES
#include "CiceroOps.cpp.inc"
