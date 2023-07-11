#include "DialectWrapper.h"
#include "MLIRGenerator.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <iostream>

#include "ASTParser.h"

using namespace std;

int main(int argc, char **argv) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<cicero_compiler::dialect::CiceroDialect>();

    if (argc != 2) {
        cout << "Usage: cicero <regex_file>" << endl;
        return -1;
    }

    auto regexAST = RegexParser::parseRegexFromFile(argv[1]);

    auto module =
        cicero_compiler::MLIRGenerator(context).mlirGen(move(regexAST));

    module.dump();
    return 0;
}