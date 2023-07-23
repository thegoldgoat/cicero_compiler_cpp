#include "AST.h"
#include "ASTParser.h"
#include <iostream>
#include <memory>
#include <vector>

using namespace RegexParser;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <input file>" << std::endl;
        return 1;
    }

    auto ast = parseRegexFromFile(argv[1]);

    std::cout << "Parsing done, printing DOT representation:" << std::endl;

    std::cout << "digraph {\n" << ast->toDotty() << "}" << std::endl;
}