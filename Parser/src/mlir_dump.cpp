#include <fstream>
#include <iostream>

#include "MLIRParser.h"
#include "RegexDialectWrapper.h"
#include "antlr4-runtime.h"
#include "regexLexer.h"
#include "regexParser.h"

using namespace std;

mlir::ModuleOp getModule(mlir::MLIRContext &context, int argc, char **argv);

void printUsages(const char *argv0) {
    cout << "[1] Usage: " << argv0 << endl;
    cout << "[2] Usage: " << argv0 << " /path/to/regexfile" << endl;
    cout << "[3] Usage: " << argv0 << " --regex <regex string>" << endl;
}

int main(int argc, char **argv) {
    mlir::MLIRContext context;

    context.getOrLoadDialect<RegexParser::dialect::RegexDialect>();

    auto module = getModule(context, argc, argv);

    if (module.verify().failed()) {
        module.dump();
        cerr << "Module (before optimization) verification failed" << endl;
        return 1;
    }

    cout << "--- MODULE BEFORE OPTIMIZATION ---\n\n";

    module.dump();

    if (RegexParser::optimizeRegex(module).failed()) {
        cerr << "Optimization failed" << endl;
        return 1;
    }

    cout << "\n\n--- MODULE AFTER OPTIMIZATION ---\n\n";

    if (module.verify().failed()) {
        module.dump();
        cerr << "Module (after optimization) verification failed" << endl;
        return 1;
    }

    module.dump();
}

mlir::ModuleOp getModule(mlir::MLIRContext &context, int argc, char **argv) {
    string regex;
    switch (argc) {
    case 1:
        cout << "Enter the regex: ";
        getline(cin, regex);
        return RegexParser::parseRegexFromString(context, regex);
        break;
    case 2:
        return RegexParser::parseRegexFromFile(context, argv[1]);
        break;
    case 3:
        if (argv[1] != string("--regex")) {
            printUsages(argv[0]);
            exit(1);
        }

        return RegexParser::parseRegexFromString(context, argv[2]);
        break;
    default:
        printUsages(argv[0]);
        exit(1);
        break;
    }
}