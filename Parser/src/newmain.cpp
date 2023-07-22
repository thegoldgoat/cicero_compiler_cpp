#include <fstream>
#include <iostream>

#include "RegexDialectWrapper.h"
#include "antlr4-runtime.h"
#include "regexLexer.h"
#include "regexParser.h"
#include "visitor/MLIRVisitor.h"

using namespace std;

mlir::ModuleOp parseRegexImpl(mlir::MLIRContext &context,
                              antlr4::ANTLRInputStream input,
                              std::string &sourceName) {
    regexLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);

    regexParser parser(&tokens);

    auto regExpRoot = parser.root();

    return RegexParser::MLIRVisitor(context, sourceName).visitRoot(regExpRoot);
}

int main(int argc, char **argv) {

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <regex file>" << endl;
        return 1;
    }

    string filename = argv[1];

    {
        ifstream stream;
        stream.open(filename);

        if (!stream) {
            return 1;
        }

        string buffer;
        cout << "--- Input ---" << endl;
        while (getline(stream, buffer)) {
            cout << buffer << endl;
        }
        cout << "---  End  ---" << endl;
    }

    ifstream stream;
    stream.open(filename);

    if (!stream) {
        return 1;
    }

    mlir::MLIRContext context;

    context.getOrLoadDialect<RegexParser::dialect::RegexDialect>();

    parseRegexImpl(context, antlr4::ANTLRInputStream(stream), filename).dump();
}