/*
 * This tool is used to measure the compile time of the Cicero compiler over a
 * set of regexes. It takes a file containing a regex per line, and measures the
 * time it takes to compile each regex. The output is the aggregated average
 * compile time.
 */
#include "cicero_helper.h"
#include "mlir/IR/MLIRContext.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "CiceroDialectWrapper.h"
#include "RegexDialectWrapper.h"

#include "MLIRParser.h"
#include "MyGreedyPass.h"

#include "CiceroMLIRGenerator.h"
#include "CiceroPasses.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

using namespace std;

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0]
             << " <regexes_file> <True if optimizations enabled, False "
                "otherwise>"
             << endl;
        return -1;
    }

    ifstream regexesFile(argv[1]);
    if (!regexesFile.is_open()) {
        cerr << "Error opening file: " << argv[1] << endl;
        return -1;
    }

    bool optimizationsEnabled;
    if (string(argv[2]) == "True") {
        optimizationsEnabled = true;
    } else if (string(argv[2]) == "False") {
        optimizationsEnabled = false;
    } else {
        cerr << "Invalid value for optimizations enabled, must be 'True' or "
                "'False', but found: "
             << argv[2] << endl;
        return -1;
    }

    vector<string> regexes;
    string line;
    while (getline(regexesFile, line)) {
        regexes.push_back(line);
    }

    ofstream outputFile("/dev/null");
    if (!outputFile.is_open()) {
        cerr << "Error opening output file: /dev/null" << endl;
        return -1;
    }

    mlir::MLIRContext context;
    context.getOrLoadDialect<cicero_compiler::dialect::CiceroDialect>();
    context.getOrLoadDialect<RegexParser::dialect::RegexDialect>();
    context.enableMultithreading(false);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto &regex : regexes) {

        auto regexModule = RegexParser::parseRegexFromString(context, regex);

        if (!regexModule) {
            cerr << "Regex parsing failed" << endl;
            return -1;
        }

        if (optimizationsEnabled) {
            if (RegexParser::optimizeRegex(regexModule, true).failed()) {
                regexModule.print(llvm::outs());
                cerr << "Regex optimization failed" << endl;
                return -1;
            }
        }

        auto module =
            cicero_compiler::CiceroMLIRGenerator(context).mlirGen(regexModule);

        mlir::RewritePatternSet patterns(&context);
        patterns.add<cicero_compiler::passes::FlattenSplit>(&context);
        patterns.add<cicero_compiler::passes::PlaceholderRemover>(&context);

        if (optimizationsEnabled) {
            patterns.add<cicero_compiler::passes::SimplifyJump>(&context);
        }

        mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

        if (runMyGreedyPass<mlir::ModuleOp>(module.getOperation(),
                                            std::move(frozenPatterns),
                                            mlir::GreedyRewriteConfig())
                .failed()) {
            module.print(llvm::outs());
            cerr << "Cicero MLIR optimization passes failed" << endl;
            return -1;
        }

        // Code generation
        cicero_compiler::dumpCompiled(module, outputFile,
                                      CiceroBinaryOutputFormat::Binary);
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time)
                        .count();

    cout << "Total time: " << duration << " microseconds" << endl;
    cout << "Average time: " << duration / regexes.size() << " microseconds"
         << endl;
}