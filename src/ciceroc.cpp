#include "CiceroDialectWrapper.h"
#include "cicero_helper.h"
#include "cicero_macros.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/ScopedHashTable.h"
#include <fstream>
#include <iostream>

#include "CiceroMLIRGenerator.h"

#include "CiceroPasses.h"
#include "cicero_const.h"

#include "MLIRParser.h"
#include "MyGreedyPass.h"

using namespace std;
namespace cl = llvm::cl;

static cl::opt<std::string> inputRegex(cl::Optional, "regex",
                                       cl::desc("<input regex>"),
                                       cl::value_desc("regex"));

static cl::opt<std::string> inputFilename(cl::Positional, cl::Optional,
                                          cl::desc("<input file>"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> outputFilename(cl::Optional,
                                           cl::desc("<output file>"), "o",
                                           cl::value_desc("filename"));

static cl::opt<enum CiceroAction>
    emitAction("emit", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpRegexMLIR, "regexmlir",
                                     "output the regex MLIR dump")),
               cl::values(clEnumValN(DumpCiceroMLIR, "ciceromlir",
                                     "output the cicero MLIR dump")),
               cl::values(clEnumValN(
                   DumpCiceroDOT, "ciceromlir.dot",
                   "output the cicero mlir operations in a graphviz format")),
               cl::values(clEnumValN(DumpCompiled, "compiled",
                                     "output the compiled artifact")));

static cl::opt<enum CiceroBinaryOutputFormat> binaryOutputFormat(
    "binary-format", cl::desc("Select the kind of binary output desired"),
    cl::values(clEnumValN(Binary, "binary", "output in binary format")),
    cl::values(clEnumValN(
        Hex, "hex", "output in hex format (one 16 bits hex value per line))")));

cl::opt<bool> optimizeAll("Oall", cl::desc("Enable all optimizations"));

cl::opt<bool>
    optimizeJumps("Ojump",
                  cl::desc("Enable optimization for jumps instructions"));

cl::opt<bool> optimizeSplitMerge(
    "Osplitmerge",
    cl::desc("Enable back-end optimization for merging splits followers"));

cl::opt<bool>
    optimizeRegex("Oregex",
                  cl::desc("Enable middle-end optimization on regex syntax"));

mlir::ModuleOp getRegexModule(mlir::MLIRContext &context);

/// @brief Output the corresponding instruction in the dot format
/// @param nodeLabel The label the node should have, e.g. "Split"
/// @param nodeColor The color the node should have
/// @param instructionIndex The address of the instruction in the program
/// instruction memory
/// @param targetIndex For jump, the jump target address. For split, the split
/// target address
/// @param linkToNext Specify if we need to link to the next instruction, true
/// for everything except for accept/accept_partial/jump/last instruction of
/// program
void outputDotFormat(const string &nodeLabel, const string &nodeColor,
                     unsigned int instructionIndex,
                     optional<unsigned int> targetIndex, bool linkToNext);

int main(int argc, char **argv) {
    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "cicero compiler\n");

    if (emitAction == None) {
        cerr << "No emit action specified (see -emit=<action>)" << endl;
        return -1;
    }

    ofstream outputFile;

    if (emitAction == DumpCompiled) {
        if (outputFilename.getNumOccurrences() == 0) {

            cerr << "No output file specified (see -o=<filename>)" << endl;
            return -1;
        }

        if (outputFilename == "-") {
            outputFile = ofstream("/dev/stdout");
        } else {
            outputFile = ofstream(outputFilename);
        }

        if (!outputFile.is_open()) {
            cerr << "Error opening output file: " << outputFilename << endl;
            return -1;
        }
    }

    mlir::MLIRContext context;
    context.getOrLoadDialect<cicero_compiler::dialect::CiceroDialect>();
    context.getOrLoadDialect<RegexParser::dialect::RegexDialect>();
    context.enableMultithreading(false);

    auto regexModule = getRegexModule(context);

    if (optimizeRegex.getValue() || optimizeAll.getValue()) {
        if (RegexParser::optimizeRegex(context, regexModule).failed()) {
            regexModule.print(llvm::outs());
            cerr << "Regex optimization failed" << endl;
            return -1;
        }
    }

    if (emitAction == DumpRegexMLIR) {
        mlir::OpPrintingFlags flags;
        regexModule.print(llvm::outs(), flags);
        return 0;
    }

    auto module =
        cicero_compiler::CiceroMLIRGenerator(context).mlirGen(regexModule);

    // Dump MLIR even before applying the patterns
    if (emitAction == DumpCiceroMLIR) {
        cout << "\n\n--- mlir before any pattern rewrites ---\n\n";
        module.print(llvm::outs());
        cout << "\n\n--- mlir after pattern rewrites      ---\n\n";
    }

    mlir::RewritePatternSet patterns(&context);
    patterns.add<cicero_compiler::passes::FlattenSplit>(&context);
    patterns.add<cicero_compiler::passes::PlaceholderRemover>(&context);

    if (optimizeJumps.getValue() || optimizeAll.getValue()) {
        patterns.add<cicero_compiler::passes::SimplifyJump>(&context);
    }

    mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (runMyGreedyPass<mlir::ModuleOp>(&context, module.getOperation(),
                                        std::move(frozenPatterns),
                                        mlir::GreedyRewriteConfig())
            .failed()) {
        module.print(llvm::outs());
        cerr << "Cicero MLIR optimization passes failed" << endl;
        return -1;
    }

    if (optimizeSplitMerge.getValue() || optimizeAll.getValue()) {
        mlir::RewritePatternSet patterns2(&context);
        patterns2.add<cicero_compiler::passes::SplitMerger>(&context);
        frozenPatterns = mlir::FrozenRewritePatternSet(std::move(patterns2));
        if (runMyGreedyPass<mlir::ModuleOp>(&context, module.getOperation(),
                                            std::move(frozenPatterns),
                                            mlir::GreedyRewriteConfig())
                .failed()) {
            module.print(llvm::outs());
            cerr << "Cicero MLIR optimization passes failed" << endl;
            return -1;
        }
    }

    if (emitAction == DumpCiceroMLIR) {
        module.print(llvm::outs());
        return 0;
    }

    if (emitAction == DumpCiceroDOT) {
        cicero_compiler::dumpCiceroDot(module);
        return 0;
    }

    // Code generation
    cicero_compiler::dumpCompiled(module, outputFile, binaryOutputFormat);

    return 0;
}

mlir::ModuleOp getRegexModule(mlir::MLIRContext &context) {

    if (inputFilename.getNumOccurrences() == 0) {
        string regex;

        if (inputRegex.getNumOccurrences() > 0) {
            regex = inputRegex.getValue();
            return RegexParser::parseRegexFromString(context, regex);
        }
        cout << "Enter regex: ";
        cin >> regex;
        cout << endl;
        return RegexParser::parseRegexFromString(context, regex);
    }

    if (inputRegex.getNumOccurrences() > 0) {
        cerr << "Cannot specify both regex and input file" << endl;
        return nullptr;
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

    return RegexParser::parseRegexFromFile(context, inputFilename);
}
