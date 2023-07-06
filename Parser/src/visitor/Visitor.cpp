// #include "Visitor.h"

#include "Visitor.h"
#include "AST.h"
#include "regexParserBaseVisitor.h"

using namespace std;

namespace RegexParser {

AST::RegExp RegexVisitor::visitRegExp(regexParser::RegExpContext *ctx) {
    vector<AST::Concatenation> concatenations;
    concatenations.reserve(ctx->concatenation().size());
    for (auto concatenation : ctx->concatenation()) {
        concatenations.push_back(visitConcatenation(concatenation));
    }

    return AST::RegExp(move(concatenations));
}

AST::Concatenation
RegexVisitor::visitConcatenation(regexParser::ConcatenationContext *ctx) {
    vector<AST::Piece> pieces;
    for (auto piece : ctx->piece()) {
        pieces.emplace_back(visitPiece(piece));
    }
    return AST::Concatenation(move(pieces));
}

AST::Piece RegexVisitor::visitPiece(regexParser::PieceContext *ctx) {
    cout << "Piece = " << ctx->getText() << endl;
    auto atom = visitAtom(ctx->atom());
    if (ctx->quantifier()) {
        return AST::Piece(move(atom), std::optional<AST::Quantifier>(
                                          visitQuantifier(ctx->quantifier())));
    } else {
        return AST::Piece(move(atom), std::optional<AST::Quantifier>());
    }
}

AST::Atom RegexVisitor::visitAtom(regexParser::AtomContext *ctx) {
    // TODO: Implement
    cout << "Atom = " << ctx->getText() << endl;
    return AST::Atom({});
}

AST::Quantifier
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

std::pair<int, int> RegexVisitor::visitQuantity(regexParser::QuantityContext *ctx) {
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