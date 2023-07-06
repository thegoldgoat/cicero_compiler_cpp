#include "AST.h"
#include "antlr4-runtime.h"
#include "regexLexer.h"
#include "regexParser.h"
#include "visitor/Visitor.h"
#include <iostream>
#include <memory>
#include <vector>

using namespace RegexParser;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <input file>" << std::endl;
        return 1;
    }
    std::ifstream stream;
    stream.open(argv[1]);

    if (!stream) {
        std::cout << "Could not open file: " << argv[1] << std::endl;
        return 1;
    }

    antlr4::ANTLRInputStream input(stream);
    regexLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);

    regexParser parser(&tokens);

    regexParser::RegExpContext *regExpTree = parser.root()->regExp();

    RegexVisitor theVisitor;
    auto ast = theVisitor.visitRegExp(regExpTree);

    std::cout << "Parsing done :)" << std::endl;
}