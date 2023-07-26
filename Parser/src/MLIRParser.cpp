#include "MLIRParser.h"
#include "ASTParser.h"
#include "MLIRRegexOptimization.h"
#include "antlr4-runtime.h"
#include "regexLexer.h"
#include "regexParser.h"
#include "visitor/MLIRVisitor.h"
#include <functional>

#include "RegexOptimizePass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace std;

namespace RegexParser {

mlir::ModuleOp parseRegexImpl(mlir::MLIRContext &context,
                              antlr4::ANTLRInputStream input,
                              const std::string &filename) {
    regexLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);

    regexParser parser(&tokens);

    auto regExpRoot = parser.root();

    return MLIRVisitor(context, filename).visitRoot(regExpRoot);
}

mlir::ModuleOp parseRegexFromFile(mlir::MLIRContext &context,
                                  const std::string &regexPath,
                                  bool printFileToStdout) {

    ifstream stream;
    stream.open(regexPath);

    if (!stream) {
        return nullptr;
    }

    if (printFileToStdout) {
        cout << "--- FILE CONTENTS ---\n";
        cout << stream.rdbuf();
        cout << "\n--- END FILE      ---\n";
    }

    stream.seekg(0, ios::beg);

    return parseRegexImpl(context, antlr4::ANTLRInputStream(stream), regexPath);
}

mlir::ModuleOp parseRegexFromString(mlir::MLIRContext &context,
                                    const std::string &regex) {
    const std::string filename = "string in memory";
    return parseRegexImpl(context, antlr4::ANTLRInputStream(regex), filename);
}

mlir::LogicalResult optimizeRegex(mlir::MLIRContext &context,
                                  mlir::ModuleOp &module) {
    mlir::PassManager pm(&context);
    mlir::OpPassManager &optimizationsPM = pm.nest<dialect::RootOp>();
    optimizationsPM.addPass(passes::createRegexOptimizePass(&context));

    return pm.run(module);
}

} // namespace RegexParser