#include "MLIRVisitor.h"
#include "RegexDialectWrapper.h"

#include "metachars.h"
#include <array>

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

    visitRegExp(root.getBody(), ctx->regExp());

    return module;
}

void MLIRVisitor::visitRegExp(mlir::Block *block,
                              regexParser::RegExpContext *ctx) {
    for (auto concatenation : ctx->concatenation()) {
        builder.setInsertionPointToEnd(block);
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

    if (ctx->DOLLAR() != nullptr) {
        builder.setInsertionPointToEnd(concatenationOp.getBody());
        auto dollarPiece =
            builder.create<dialect::PieceOp>(LOCATION_MACRO(ctx->DOLLAR()));
        builder.setInsertionPointToStart(dollarPiece.getBody());
        builder.create<dialect::DollarOp>(LOCATION_MACRO(ctx->DOLLAR()));
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
    // Metachar
    if (ctx->metachar()) {
        builder.create<dialect::GroupOp>(LOCATION_MACRO(ctx),
                                         visitMetachar(ctx->metachar()));
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
        visitRegExp(subregex.getBody(), ctx->regExp());
        return;
    }

    // Group
    if (ctx->LBRACKET()) {
        bool charSet[256];
        memset(charSet, false, sizeof(charSet));
        for (auto &groupCtx : ctx->group()) {
            if (groupCtx->group_metachar()) {
                auto mergeChar = visitGroupMetachar(groupCtx->group_metachar());

                for (std::size_t i = 0; i < sizeof(charSet); i++) {
                    charSet[i] = charSet[i] || mergeChar[i];
                }
            } else if (groupCtx->single_char) {
                charSet[static_cast<int>(groupCtx->single_char->getText()[0])] =
                    true;
            } else {
                auto begin = groupCtx->first_char->getText()[0];
                auto end = groupCtx->second_char->getText()[0];

                if (begin > end) {
                    throw std::runtime_error(
                        "Invalid character range, second must be greater than "
                        "first: '" +
                        groupCtx->getText() + "' at " +
                        groupCtx->getSourceInterval().toString());
                }

                for (int i = begin; i <= end; i++) {
                    charSet[i] = true;
                }
            }
        }

        // Negate the character set if the HAT is present
        if (ctx->GROUP_HAT()) {
            for (std::vector<bool>::size_type i = 0; i < sizeof(charSet); i++) {
                charSet[i] = !charSet[i];
            }
        }

        builder.create<dialect::GroupOp>(
            LOCATION_MACRO(ctx),
            builder.getDenseBoolArrayAttr(
                llvm::ArrayRef<bool>(charSet, sizeof(charSet))));
        return;
    }

    throw std::runtime_error("Invalid atom: " + ctx->getText() + " at " +
                             ctx->getSourceInterval().toString());
}

void MLIRVisitor::visitQuantifier(regexParser::QuantifierContext *ctx) {
    if (ctx->QUESTION()) {
        builder.create<dialect::QuantifierOp>(LOCATION_MACRO(ctx), 0, 1);
        return;
    }

    if (ctx->STAR()) {
        builder.create<dialect::QuantifierOp>(LOCATION_MACRO(ctx), 0, -1);
        return;
    }

    if (ctx->PLUS()) {
        builder.create<dialect::QuantifierOp>(LOCATION_MACRO(ctx), 1, -1);
        return;
    }

    if (ctx->quantity()) {
        auto boundaries = visitQuantity(ctx->quantity());

        builder.create<dialect::QuantifierOp>(
            LOCATION_MACRO(ctx), boundaries.first, boundaries.second);
        return;
    }

    throw std::runtime_error("Invalid quantifier: " + ctx->getText() + " at " +
                             ctx->getSourceInterval().toString());
}

#define BUILD_BOOL_REF_ARRAY_MACRO(array)                                      \
    builder.getDenseBoolArrayAttr(llvm::ArrayRef<bool>(array, sizeof(array)))

mlir::DenseBoolArrayAttr MLIRVisitor::getMetacharArray(char metachar) {
    switch (metachar) {
    case 'd':
        return BUILD_BOOL_REF_ARRAY_MACRO(DIGIT_SET);
    case 'D':
        return BUILD_BOOL_REF_ARRAY_MACRO(DIGIT_SET_COMPLEMENTED);
    case 'w':
        return BUILD_BOOL_REF_ARRAY_MACRO(WORD_SET);
    case 'W':
        return BUILD_BOOL_REF_ARRAY_MACRO(WORD_SET_COMPLEMENTED);
    case 's':
        return BUILD_BOOL_REF_ARRAY_MACRO(WHITESPACE_SET);
    case 'S':
        return BUILD_BOOL_REF_ARRAY_MACRO(WHITESPACE_SET_COMPLEMENTED);
    default:
        throw std::exception();
    }
}

mlir::DenseBoolArrayAttr
MLIRVisitor::visitMetachar(regexParser::MetacharContext *ctx) {
    try {
        return getMetacharArray(ctx->getText()[1]);
    } catch (std::exception e) {
        throw std::runtime_error("Invalid metachar: " + ctx->getText() +
                                 " at " + ctx->getSourceInterval().toString());
    }
}

mlir::DenseBoolArrayAttr
MLIRVisitor::visitGroupMetachar(regexParser::Group_metacharContext *ctx) {
    try {
        return getMetacharArray(ctx->getText()[1]);
    } catch (std::exception e) {
        throw std::runtime_error("Invalid metachar: " + ctx->getText() +
                                 " at " + ctx->getSourceInterval().toString());
    }
}

std::pair<int, int>
MLIRVisitor::visitQuantity(regexParser::QuantityContext *ctx) {
    if (ctx->exactlynum) {
        int exactVal = stoi(ctx->exactlynum->getText());
        return {exactVal, exactVal};
    } else if (ctx->atleastnum) {
        int atleastVal = stoi(ctx->atleastnum->getText());
        return {atleastVal, -1};
    } else {
        int minVal = stoi(ctx->minnum->getText());
        int maxVal = stoi(ctx->maxnum->getText());
        return {minVal, maxVal};
    }
}

} // namespace RegexParser
