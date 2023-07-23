#include <fstream>
#include <iostream>

#include "RegexDialectWrapper.h"
#include "antlr4-runtime.h"
#include "regexLexer.h"
#include "regexParser.h"
#include "MLIRParser.h"

using namespace std;

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

    mlir::MLIRContext context;

    context.getOrLoadDialect<RegexParser::dialect::RegexDialect>();

    RegexParser::parseRegexFromFile(context, filename).dump();
}