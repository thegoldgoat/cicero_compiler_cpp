#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "regexParserBaseVisitor.h"

namespace RegexParser {

class MLIRVisitor {
  public:
    MLIRVisitor(mlir::MLIRContext &context, std::string &filename)
        : builder(&context) {
        this->filename = builder.getStringAttr(filename);
    };

    mlir::ModuleOp visitRoot(regexParser::RootContext *ctx);

    void visitRegExp(regexParser::RegExpContext *ctx);

    void visitConcatenation(regexParser::ConcatenationContext *ctx);

    void visitPiece(regexParser::PieceContext *ctx);

    void visitAtom(regexParser::AtomContext *ctx);

    void visitQuantifier(regexParser::QuantifierContext *ctx);
    //
    //    vector<bool> visitMetachar(regexParser::MetacharContext *ctx);
    //
    //    pair<int, int> visitQuantity(regexParser::QuantityContext *ctx);

  private:
    mlir::OpBuilder builder;

    mlir::StringAttr filename;
};

} // namespace RegexParser
