#include "MLIRParser.h"
#include "ASTParser.h"
#include "MLIRRegexOptimization.h"
#include "antlr4-runtime.h"
#include "regexLexer.h"
#include "regexParser.h"
#include "visitor/MLIRVisitor.h"
#include <functional>

#include "MyGreedyPass.h"
#include "mlir/Transforms/Passes.h"

using namespace std;

namespace RegexParser {

mlir::ModuleOp parseRegexImpl(mlir::MLIRContext &context,
                              antlr4::ANTLRInputStream input,
                              const std::string &filename) {
    regexLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);

    regexParser parser(&tokens);

    parser.setErrorHandler(std::make_shared<antlr4::BailErrorStrategy>());

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

mlir::LogicalResult optimizeRegex(mlir::ModuleOp &module,
                                  bool optimizeBoundaries) {
    mlir::MLIRContext *context = module.getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<passes::FactorizeRoot, passes::FactorizeSubregex,
                 passes::SimplifySubregexNotQuantified,
                 passes::SimplifySubregexSinglePiece>(context);

    if (optimizeBoundaries) {
        patterns.add<passes::SimplifyLeadingQuantifiers>(context);
    }

    return runMyGreedyPass<mlir::ModuleOp>(
        module.getOperation(),
        mlir::FrozenRewritePatternSet(std::move(patterns)),
        mlir::GreedyRewriteConfig());
}

} // namespace RegexParser