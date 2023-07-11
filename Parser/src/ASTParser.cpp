#include "ASTParser.h"
#include "antlr4-runtime.h"
#include "regexLexer.h"
#include "regexParser.h"
#include "visitor/Visitor.h"

using namespace std;

namespace RegexParser {

unique_ptr<AST::RegExp> parseRegexFromFile(const string &regexPath) {
    ifstream stream;
    stream.open(regexPath);

    if (!stream) {
        return nullptr;
    }

    antlr4::ANTLRInputStream input(stream);
    regexLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);

    regexParser parser(&tokens);

    regexParser::RegExpContext *regExpTree = parser.root()->regExp();

    return move(RegexVisitor().visitRegExp(regExpTree));
}
} // namespace RegexParser