#pragma once

#include "AST.h"
#include "regexParserBaseVisitor.h"
#include <memory>
#include <string>

namespace RegexParser {

using namespace std;

class RegexVisitor {
  public:
    unique_ptr<AST::Root> visitRoot(regexParser::RootContext *ctx);

    unique_ptr<AST::RegExp> visitRegExp(regexParser::RegExpContext *ctx);

    unique_ptr<AST::Concatenation>
    visitConcatenation(regexParser::ConcatenationContext *ctx);

    unique_ptr<AST::Piece> visitPiece(regexParser::PieceContext *ctx);

    unique_ptr<AST::Atom> visitAtom(regexParser::AtomContext *ctx);

    unique_ptr<AST::Quantifier>
    visitQuantifier(regexParser::QuantifierContext *ctx);

    vector<bool> visitMetachar(regexParser::MetacharContext *ctx);
    vector<bool> visitGroupMetachar(regexParser::Group_metacharContext *ctx);

    pair<int, int> visitQuantity(regexParser::QuantityContext *ctx);

  private:
    vector<bool> getMetacharArray(char metachar);
};

} // namespace RegexParser