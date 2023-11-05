#include "RegexDialectWrapper.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace RegexParser::dialect;

#include "RegexDialect.cpp.inc"

void RegexDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "RegexOps.cpp.inc"
        >();
}

#define GET_OP_CLASSES
#include "RegexOps.cpp.inc"

#include "MatchCharHelper.h"

DEFINE_MATCH_CHAR_PARSER_MACRO(MatchCharOp)
DEFINE_MATCH_CHAR_PRINTER_MACRO(MatchCharOp)
DEFINE_MATCH_CHAR_PARSER_MACRO(NotMatchCharOp)
DEFINE_MATCH_CHAR_PRINTER_MACRO(NotMatchCharOp)