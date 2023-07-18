#include "ASTParser.h"
#include "antlr4-runtime.h"
#include "regexLexer.h"
#include "regexParser.h"
#include "visitor/Visitor.h"
#include <functional>

using namespace std;

namespace RegexParser {

unique_ptr<AST::Root> parseRegexImpl(antlr4::ANTLRInputStream input) {
    regexLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);

    regexParser parser(&tokens);

    auto regExpRoot = parser.root();
    
    return move(RegexVisitor().visitRoot(regExpRoot));
}

unique_ptr<AST::Root> parseRegexFromFile(const string &regexPath) {

    ifstream stream;
    stream.open(regexPath);

    if (!stream) {
        return nullptr;
    }
    return parseRegexImpl(antlr4::ANTLRInputStream(stream));
}

std::unique_ptr<AST::Root> parseRegexFromString(const std::string &regex) {
    return parseRegexImpl(antlr4::ANTLRInputStream(regex));
}
} // namespace RegexParser