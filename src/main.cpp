#include "DialectWrapper.h"
#include "MLIRGenerator.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <fstream>
#include <iostream>

#include "ASTParser.h"

using namespace std;
namespace cl = llvm::cl;

static cl::opt<std::string> inputRegex(cl::Optional, "regex",
                                       cl::desc("<input regex>"),
                                       cl::value_desc("regex"));

static cl::opt<std::string> inputFilename(cl::Positional, cl::Optional,
                                          cl::desc("<input file>"),
                                          cl::value_desc("filename"));

unique_ptr<RegexParser::AST::RegExp> getAST() {

    if (inputFilename.getNumOccurrences() == 0) {
        string regex;

        if (inputRegex.getNumOccurrences() > 0) {
            regex = inputRegex;
            return RegexParser::parseRegexFromString(regex);
        }
        cout << "Enter regex: ";
        cin >> regex;
        cout << endl;
        return RegexParser::parseRegexFromString(regex);
    }

    ifstream regexFile(inputFilename);
    if (!regexFile.is_open()) {
        cerr << "Error opening file: " << inputFilename << endl;
        return nullptr;
    }

    cout << "--- Regex file content  ---" << endl;
    // Read the file content and print to std::cout
    string line;
    while (getline(regexFile, line)) {
        cout << line << endl;
    }
    cout << "--- End of file content ---" << endl;

    return RegexParser::parseRegexFromFile(inputFilename);
}

int main(int argc, char **argv) {
    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "cicero compiler\n");

    auto regexAST = getAST();

    if (!regexAST) {
        cerr << "Error parsing regex?" << endl;
        return -1;
    }

    mlir::MLIRContext context;
    context.getOrLoadDialect<cicero_compiler::dialect::CiceroDialect>();

    context.enableMultithreading(false);

    auto module =
        cicero_compiler::MLIRGenerator(context).mlirGen(move(regexAST));

    if (mlir::failed(mlir::verify(module))) {
        module.print(llvm::outs());
        module.emitError("module verification error");
        return -1;
    }

    mlir::PassManager pm(&context);
    applyPassManagerCLOptions(pm);

    pm.addPass(mlir::createCanonicalizerPass());

    if (mlir::failed(pm.run(module))) {
        module.print(llvm::outs());
        cerr << "Error running canonicalizer pass" << endl;
        return -1;
    }

    module.print(llvm::outs());
    return 0;
}