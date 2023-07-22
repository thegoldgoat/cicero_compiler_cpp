#include "MLIRVisitor.h"
#include "RegexDialectWrapper.h"

namespace RegexParser {

#define LOCATION_MACRO(ctx_macro)                                              \
    mlir::FileLineColLoc::get(filename, 0, ctx_macro->getSourceInterval().a)

mlir::ModuleOp MLIRVisitor::visitRoot(regexParser::RootContext *ctx) {
    auto module = mlir::ModuleOp::create(LOCATION_MACRO(ctx));

    bool hasPrefix = ctx->noprefix == nullptr;
    bool hasSuffix = ctx->nosuffix == nullptr;

    builder.setInsertionPointToStart(module.getBody());

    auto root = builder.create<dialect::RootOp>(LOCATION_MACRO(ctx), hasPrefix,
                                                hasSuffix);

    builder.setInsertionPointToStart(root.getBody());

    visitRegExp(ctx->regExp());

    return module;
}

void MLIRVisitor::visitRegExp(regexParser::RegExpContext *ctx) {
    auto alternationOp =
        builder.create<dialect::AlternationOp>(LOCATION_MACRO(ctx));
    for (auto concatenation : ctx->concatenation()) {
        builder.setInsertionPointToEnd(alternationOp.getBody());
        visitConcatenation(concatenation);
    }
}

void MLIRVisitor::visitConcatenation(regexParser::ConcatenationContext *ctx) {
    auto concatenationOp =
        builder.create<dialect::ConcatenationOp>(LOCATION_MACRO(ctx));
    for (auto piece : ctx->piece()) {
        builder.setInsertionPointToEnd(concatenationOp.getBody());
        visitPiece(piece);
    }
}

void MLIRVisitor::visitPiece(regexParser::PieceContext *ctx) {
    auto pieceOp = builder.create<dialect::PieceOp>(LOCATION_MACRO(ctx));
    builder.setInsertionPointToStart(pieceOp.getBody());
    visitAtom(ctx->atom());
    if (ctx->quantifier()) {
        builder.setInsertionPointToEnd(pieceOp.getBody());
        visitQuantifier(ctx->quantifier());
    }
}

void MLIRVisitor::visitAtom(regexParser::AtomContext *ctx) {
    // TODO: Metachar
    if (ctx->metachar()) {
        builder.create<dialect::GroupOp>(LOCATION_MACRO(ctx));
        return;
    }

    // Single char
    if (ctx->terminal_sequence()) {
        builder.create<dialect::MatchCharOp>(
            LOCATION_MACRO(ctx), ctx->terminal_sequence()->getText()[0]);
        return;
    }

    // Any char
    if (ctx->ANYCHAR()) {
        builder.create<dialect::MatchAnyCharOp>(LOCATION_MACRO(ctx));
        return;
    }

    // Subregex
    if (ctx->LPAR()) {
        auto subregex =
            builder.create<dialect::SubRegexOp>(LOCATION_MACRO(ctx));
        builder.setInsertionPointToStart(subregex.getBody());
        visitRegExp(ctx->regExp());
        return;
    }

    // TODO: Group
    if (ctx->LBRACKET()) {
        builder.create<dialect::GroupOp>(LOCATION_MACRO(ctx));
        return;
    }

    throw std::runtime_error("Invalid atom: " + ctx->getText() + " at " +
                             ctx->getSourceInterval().toString());
}

void MLIRVisitor::visitQuantifier(regexParser::QuantifierContext *ctx) {
    builder.create<dialect::QuantifierOp>(LOCATION_MACRO(ctx), 10, 100);
}

} // namespace RegexParser
