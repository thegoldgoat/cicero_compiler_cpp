// #include "Visitor.h"

#include "Visitor.h"
#include "AST.h"
#include "metachars.h"
#include "regexParserBaseVisitor.h"

using namespace std;

namespace RegexParser {
unique_ptr<AST::Root> RegexVisitor::visitRoot(regexParser::RootContext *ctx) {
    auto regExp = visitRegExp(ctx->regExp());
    bool hasPrefix = ctx->noprefix == nullptr;
    bool hasSuffix = ctx->nosuffix == nullptr;
    return make_unique<AST::Root>(move(regExp), hasPrefix, hasSuffix);
}

unique_ptr<AST::RegExp>
RegexVisitor::visitRegExp(regexParser::RegExpContext *ctx) {
    vector<unique_ptr<AST::Concatenation>> concatenations;
    concatenations.reserve(ctx->concatenation().size());
    for (auto concatenation : ctx->concatenation()) {
        concatenations.push_back(visitConcatenation(concatenation));
    }

    return make_unique<AST::RegExp>(move(concatenations));
}

unique_ptr<AST::Concatenation>
RegexVisitor::visitConcatenation(regexParser::ConcatenationContext *ctx) {
    vector<unique_ptr<AST::Piece>> pieces;
    pieces.reserve(ctx->piece().size());
    for (auto piece : ctx->piece()) {
        pieces.emplace_back(visitPiece(piece));
    }
    return make_unique<AST::Concatenation>(AST::Concatenation(move(pieces)));
}

unique_ptr<AST::Piece>
RegexVisitor::visitPiece(regexParser::PieceContext *ctx) {
    auto atom = visitAtom(ctx->atom());
    if (ctx->quantifier()) {
        return make_unique<AST::Piece>(move(atom),
                                       visitQuantifier(ctx->quantifier()));
    } else {
        return make_unique<AST::Piece>(move(atom), nullptr);
    }
}

unique_ptr<AST::Atom> RegexVisitor::visitAtom(regexParser::AtomContext *ctx) {
    // Metachar
    if (ctx->metachar()) {
        auto charSet = visitMetachar(ctx->metachar());
        return make_unique<AST::Group>(AST::Group(move(charSet)));
    }

    // Single char
    if (ctx->terminal_sequence()) {
        return make_unique<AST::Char>(
            AST::Char({ctx->terminal_sequence()->getText()[0]}));
    }

    // Any char
    if (ctx->ANYCHAR()) {
        return make_unique<AST::AnyChar>(AST::AnyChar());
    }

    // Subregex
    if (ctx->LPAR()) {
        auto subregex = visitRegExp(ctx->regExp());
        return make_unique<AST::SubRegex>(AST::SubRegex(move(subregex)));
    }

    // Group
    if (ctx->LBRACKET()) {
        vector<bool> charSet(256, false);
        for (auto &groupCtx : ctx->group()) {
            if (groupCtx->metachar()) {
                auto mergeChar = visitMetachar(groupCtx->metachar());

                for (vector<bool>::size_type i = 0; i < charSet.size(); i++) {
                    charSet[i] = charSet[i] || mergeChar[i];
                }
            } else if (groupCtx->single_char) {
                charSet[groupCtx->single_char->getText()[0]] = true;
            } else {
                auto begin = groupCtx->first_char->getText()[0];
                auto end = groupCtx->second_char->getText()[0];

                if (begin > end) {
                    throw runtime_error(
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
        if (ctx->HAT()) {
            for (vector<bool>::size_type i = 0; i < charSet.size(); i++) {
                charSet[i] = !charSet[i];
            }
        }

        return make_unique<AST::Group>(AST::Group(move(charSet)));
    }

    throw runtime_error("Invalid atom: " + ctx->getText() + " at " +
                        ctx->getSourceInterval().toString());
}

unique_ptr<AST::Quantifier>
RegexVisitor::visitQuantifier(regexParser::QuantifierContext *ctx) {
    if (ctx->QUESTION()) {
        return AST::Quantifier::buildOptionalQuantifier();
    }

    if (ctx->STAR()) {
        return AST::Quantifier::buildStarQuantifier();
    }

    if (ctx->PLUS()) {
        return AST::Quantifier::buildPlusQuantifier();
    }

    if (ctx->quantity()) {
        auto boundaries = visitQuantity(ctx->quantity());

        return AST::Quantifier::buildRangeQuantifier(boundaries.first,
                                                     boundaries.second);
    }

    throw runtime_error("Invalid quantifier");
}

vector<bool> RegexVisitor::visitMetachar(regexParser::MetacharContext *ctx) {
    bool *selectedBuffer;
    char metachar = ctx->getText()[1];
    switch (metachar) {
    case 'd':
        selectedBuffer = DIGIT_SET;
        break;
    case 'D':
        selectedBuffer = DIGIT_SET_COMPLEMENTED;
        break;
    case 'w':
        selectedBuffer = WORD_SET;
        break;
    case 'W':
        selectedBuffer = WORD_SET_COMPLEMENTED;
        break;
    case 's':
        selectedBuffer = WHITESPACE_SET;
        break;
    case 'S':
        selectedBuffer = WHITESPACE_SET_COMPLEMENTED;
        break;
    default:
        throw runtime_error("Invalid metachar");
    }
    vector<bool> retVal(256);
    for (vector<bool>::size_type i = 0; i < retVal.size(); i++) {
        retVal[i] = selectedBuffer[i];
    }

    return retVal;
}

pair<int, int> RegexVisitor::visitQuantity(regexParser::QuantityContext *ctx) {
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