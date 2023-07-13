#include "DialectWrapper.h"
#include "MLIRGenerator.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <fstream>
#include <iostream>

#include "ASTParser.h"

using namespace std;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::value_desc("filename"));

int main(int argc, char **argv) {

    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "cicero compiler\n");

    mlir::MLIRContext context;
    context.getOrLoadDialect<cicero_compiler::dialect::CiceroDialect>();

    // Open the file and print to output
    ifstream regexFile(inputFilename);
    if (!regexFile.is_open()) {
        cerr << "Error opening file: " << inputFilename << endl;
        return -1;
    }

    cout << "--- Regex file content  ---" << endl;
    // Read the file content and print to std::cout
    string line;
    while (getline(regexFile, line)) {
        cout << line << endl;
    }
    cout << "--- End of file content ---" << endl;

    auto regexAST = RegexParser::parseRegexFromFile(inputFilename);

    if (!regexAST) {
        cerr << "Error parsing regex? Maybe the file does not exists or it is "
                "incorrect?"
             << endl;
        return -1;
    }

    context.enableMultithreading(false);

    auto module =
        cicero_compiler::MLIRGenerator(context).mlirGen(move(regexAST));


    if (mlir::failed(mlir::verify(module))) {
        module.print(llvm::outs());
        module.emitError("module verification error");
        return -1;
    }

    mlir::PassManager pm(&context);

    pm.addPass(mlir::createCanonicalizerPass());

    // pm.addNestedPass<cicero_compiler::dialect::PlaceholderOp>(
    //     mlir::createCanonicalizerPass());

    pm.enableStatistics(mlir::PassDisplayMode::Pipeline);
    pm.enableCrashReproducerGeneration("./reproducer.mlir", false);
    if (mlir::failed(pm.run(module))) {
        module.print(llvm::outs());
        cerr << "Error running canonicalizer pass" << endl;
        return -1;
    }

    module.print(llvm::outs());
    return 0;
}