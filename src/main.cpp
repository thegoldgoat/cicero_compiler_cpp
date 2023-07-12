#include "DialectWrapper.h"
#include "MLIRGenerator.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <fstream>
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

    // Open the file and print to output
    ifstream regexFile(argv[1]);
    if (!regexFile.is_open()) {
        cout << "Error opening file: " << argv[1] << endl;
        return -1;
    }

    cout << "--- Regex file content  ---" << endl;
    // Read the file content and print to std::cout
    string line;
    while (getline(regexFile, line)) {
        cout << line << endl;
    }
    cout << "--- End of file content ---" << endl;

    auto regexAST = RegexParser::parseRegexFromFile(argv[1]);

    if (!regexAST) {
        cout << "Error parsing regex? Maybe the file does not exists or it is "
                "incorrect?"
             << endl;
        return -1;
    }

    auto module =
        cicero_compiler::MLIRGenerator(context).mlirGen(move(regexAST));

    module.print(llvm::outs());
    return 0;
}