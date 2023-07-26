#pragma once

#define DEFINE_MATCH_CHAR_PRINTER_MACRO(OpName)                                \
    void OpName::print(mlir::OpAsmPrinter &printer) {                          \
        printer.printOptionalAttrDict((*this)->getAttrs(),                     \
                                      /*elidedAttrs=*/{"targetChar"});         \
        printer << " ";                                                        \
        printer << getTargetChar();                                            \
    }

#define DEFINE_MATCH_CHAR_PARSER_MACRO(OpName)                                 \
    mlir::ParseResult OpName::parse(mlir::OpAsmParser &parser,                 \
                                    mlir::OperationState &result) {            \
        mlir::IntegerAttr targetChar;                                          \
        if (parser.parseOptionalAttrDict(result.attributes) ||                 \
            parser.parseAttribute(targetChar, "targetChar",                    \
                                  result.attributes))                          \
            return failure();                                                  \
        result.addTypes(targetChar.getType());                                 \
        return success();                                                      \
    }
