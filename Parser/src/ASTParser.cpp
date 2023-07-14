#include "ASTParser.h"
#include "antlr4-runtime.h"
#include "regexLexer.h"
#include "regexParser.h"
#include "visitor/Visitor.h"
#include <functional>

using namespace std;

namespace RegexParser {

unique_ptr<AST::RegExp> parseRegexImpl(antlr4::ANTLRInputStream input) {
    regexLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);

    regexParser parser(&tokens);

    regexParser::RegExpContext *regExpTree = parser.root()->regExp();

    return move(RegexVisitor().visitRegExp(regExpTree));
}

unique_ptr<AST::RegExp> parseRegexFromFile(const string &regexPath) {

    ifstream stream;
    stream.open(regexPath);

    if (!stream) {
        return nullptr;
    }
    return parseRegexImpl(antlr4::ANTLRInputStream(stream));
}

std::unique_ptr<AST::RegExp> parseRegexFromString(const std::string &regex) {
    return parseRegexImpl(antlr4::ANTLRInputStream(regex));
}
} // namespace RegexParser