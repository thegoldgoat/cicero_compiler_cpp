#include "MLIRParser.h"
#include "ASTParser.h"
#include "MLIRRegexOptimization.h"
#include "antlr4-runtime.h"
#include "regexLexer.h"
#include "regexParser.h"
#include "visitor/MLIRVisitor.h"
#include <functional>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
        cout << "--- END FILE      ---\n";
    }

    stream.seekg(0, ios::beg);

    return parseRegexImpl(context, antlr4::ANTLRInputStream(stream), regexPath);
}

mlir::ModuleOp parseRegexFromString(mlir::MLIRContext &context,
                                    const std::string &regex) {
    const std::string filename = "string in memory";
    return parseRegexImpl(context, antlr4::ANTLRInputStream(regex), filename);
}
} // namespace RegexParser

mlir::LogicalResult RegexParser::optimizeRegex(mlir::ModuleOp &module) {
    mlir::RewritePatternSet patterns(module.getContext());
    patterns.add<passes::FactorizeRoot, passes::FactorizeSubregex>(
        module.getContext());

    mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    return mlir::applyPatternsAndFoldGreedily(module, frozenPatterns);
}