#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "regexParserBaseVisitor.h"
#include <memory>
#include <vector>

namespace RegexParser {

class MLIRVisitor {
  public:
    MLIRVisitor(mlir::MLIRContext &context, const std::string &filename)
        : builder(&context) {
        this->filename = builder.getStringAttr(filename);
    };

    mlir::ModuleOp visitRoot(regexParser::RootContext *ctx);

    void visitRegExp(regexParser::RegExpContext *ctx);

    void visitConcatenation(regexParser::ConcatenationContext *ctx);

    void visitPiece(regexParser::PieceContext *ctx);

    void visitAtom(regexParser::AtomContext *ctx);

    void visitQuantifier(regexParser::QuantifierContext *ctx);

    mlir::DenseBoolArrayAttr visitMetachar(regexParser::MetacharContext *ctx);

    std::pair<int, int> visitQuantity(regexParser::QuantityContext *ctx);

  private:
    mlir::OpBuilder builder;

    mlir::StringAttr filename;
};

} // namespace RegexParser
