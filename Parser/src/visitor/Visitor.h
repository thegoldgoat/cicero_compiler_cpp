#pragma once

#include "regexParserBaseVisitor.h"
#include <string>
#include <memory>
#include "AST.h"

namespace RegexParser {

class RegexVisitor {
  public:
    AST::RegExp visitRegExp(regexParser::RegExpContext *ctx);

    AST::Concatenation visitConcatenation(regexParser::ConcatenationContext *ctx);

    AST::Piece visitPiece(regexParser::PieceContext *ctx);

    AST::Atom visitAtom(regexParser::AtomContext *ctx);

    AST::Quantifier visitQuantifier(regexParser::QuantifierContext *ctx);

    std::pair<int, int> visitQuantity(regexParser::QuantityContext *ctx);
};

}