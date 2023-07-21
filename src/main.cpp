#include "DialectWrapper.h"
#include "MLIRGenerator.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ScopedHashTable.h"
#include <fstream>
#include <iostream>

#include "cicero_const.h"

#include "ASTParser.h"

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
               cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
               cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
               cl::values(clEnumValN(
                   DumpDOT, "mlir.dot",
                   "output the cicero instruction in a graphviz format")),
               cl::values(clEnumValN(DumpCompiled, "compiled",
                                     "output the compiled artifact")));

static cl::opt<enum CiceroBinaryOutputFormat> binaryOutputFormat(
    "binary-format", cl::desc("Select the kind of binary output desired"),
    cl::values(clEnumValN(Binary, "binary", "output in binary format")),
    cl::values(clEnumValN(
        Hex, "hex", "output in hex format (one 16 bits hex value per line))")));

unique_ptr<RegexParser::AST::Root> getAST();

#define CAST_MACRO(resultName, inputOperation, operationType)                  \
    auto resultName = mlir::dyn_cast<operationType>(inputOperation)

void outputToFileBinaryFormat(uint16_t opCode, uint16_t opData,
                              ofstream &outputStream);
void outputToFileHexFormat(uint16_t opCode, uint16_t opData,
                           ofstream &outputStream);

/// @brief Output the corresponding instruction in the dot format
/// @param instructionName The name of the instruction e.g. MatchChar
/// @param instructionArgument The argument (data) of the instruction e.g. for
/// MatchChar argument is the char to match
/// @param instructionIndex The address of the instruction in the program
/// instruction memory
/// @param targetIndex For jump, the jump target address. For split, the split
/// target address
/// @param linkToNext Specify if we need to link to the next instruction, true
/// for everything except for accept/accept_partial/jump/last instruction of
/// program
void outputDotFormat(string instructionName,
                     optional<string> instructionArgument,
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

        outputFile = ofstream(outputFilename);

        if (!outputFile.is_open()) {
            cerr << "Error opening output file: " << outputFilename << endl;
            return -1;
        }
    }

    auto regexAST = getAST();

    if (!regexAST) {
        cerr << "Error parsing regex?" << endl;
        return -1;
    }

    if (emitAction == DumpAST) {
        std::cout << "digraph {\n" << regexAST->toDotty() << "}" << std::endl;
        return 0;
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

    if (emitAction == DumpMLIR) {
        module.print(llvm::outs());
        return 0;
    }

    // Code generation
    llvm::ScopedHashTable<mlir::StringRef, unsigned int> symbolTable;
    llvm::ScopedHashTableScope<mlir::StringRef, unsigned int> scopedTable(
        symbolTable);
    unsigned int operationIndex = 0;
    module.getBody()->walk(
        [&symbolTable, &operationIndex](mlir::Operation *op) {
            auto opSymbol = mlir::SymbolTable::getSymbolName(op);
            if (opSymbol) {
                symbolTable.insert(opSymbol, operationIndex);
            }
            operationIndex++;
        });

    operationIndex = 0;
    if (emitAction == DumpDOT) {

        cout << "digraph {" << endl;
        module.getBody()->walk([&symbolTable,
                                &operationIndex](mlir::Operation *op) {
            bool isLastInstuction = op->getNextNode() == nullptr;
            if (CAST_MACRO(matchCharOp, op,
                           cicero_compiler::dialect::MatchCharOp)) {
                outputDotFormat("MatchChar",
                                string(1, matchCharOp.getTargetChar()),
                                operationIndex, nullopt, !isLastInstuction);
            } else if (CAST_MACRO(notMatchOp, op,
                                  cicero_compiler::dialect::NotMatchCharOp)) {
                outputDotFormat("NotMatchChar",
                                string(1, notMatchOp.getTargetChar()),
                                operationIndex, nullopt, !isLastInstuction);
            } else if (CAST_MACRO(matchAnyOp, op,
                                  cicero_compiler::dialect::MatchAnyOp)) {
                outputDotFormat("MatchAny", nullopt, operationIndex, nullopt,
                                !isLastInstuction);
            } else if (CAST_MACRO(flatSplitOp, op,
                                  cicero_compiler::dialect::FlatSplitOp)) {
                uint16_t splitTargetIndex =
                    symbolTable.lookup(flatSplitOp.getSplitTarget());

                if (isLastInstuction) {
                    throw std::runtime_error(
                        "Last instruction of program cannot be a split, how "
                        "did we get here???");
                }

                outputDotFormat("Split", nullopt, operationIndex,
                                splitTargetIndex, true);

            } else if (CAST_MACRO(acceptOp, op,
                                  cicero_compiler::dialect::AcceptOp)) {
                outputDotFormat("Accept", nullopt, operationIndex, nullopt,
                                false);
            } else if (CAST_MACRO(acceptPartialOp, op,
                                  cicero_compiler::dialect::AcceptPartialOp)) {
                outputDotFormat("AcceptPartial", nullopt, operationIndex,
                                nullopt, false);
            } else if (CAST_MACRO(jumpOp, op,
                                  cicero_compiler::dialect::JumpOp)) {
                uint16_t jumpTargetIndex =
                    symbolTable.lookup(jumpOp.getTarget());

                outputDotFormat("Jump", nullopt, operationIndex,
                                jumpTargetIndex, false);
            } else {
                throw std::runtime_error(
                    "Graphviz output for operation not implemented: " +
                    op->getName().getStringRef().str());
            }
            operationIndex++;
        });

        cout << "}" << endl;
        return 0;
    }

    auto writeFunction = binaryOutputFormat == Hex ? outputToFileHexFormat
                                                   : outputToFileBinaryFormat;
    module.getBody()->walk([&writeFunction, &outputFile, &symbolTable,
                            &operationIndex](mlir::Operation *op) {
        // Try to cast to concrete operations
        if (CAST_MACRO(matchCharOp, op,
                       cicero_compiler::dialect::MatchCharOp)) {
            writeFunction(CiceroOpCodes::MATCH_CHAR,
                          matchCharOp.getTargetChar(), outputFile);
        } else if (CAST_MACRO(notMatchOp, op,
                              cicero_compiler::dialect::NotMatchCharOp)) {
            writeFunction(CiceroOpCodes::NOT_MATCH_CHAR,
                          notMatchOp.getTargetChar(), outputFile);
        } else if (CAST_MACRO(matchAnyOp, op,
                              cicero_compiler::dialect::MatchAnyOp)) {
            writeFunction(CiceroOpCodes::MATCH_ANY, 0, outputFile);
        } else if (CAST_MACRO(flatSplitOp, op,
                              cicero_compiler::dialect::FlatSplitOp)) {
            uint16_t splitTargetIndex =
                symbolTable.lookup(flatSplitOp.getSplitTarget());

            writeFunction(CiceroOpCodes::SPLIT, splitTargetIndex, outputFile);
        } else if (CAST_MACRO(acceptOp, op,
                              cicero_compiler::dialect::AcceptOp)) {
            writeFunction(CiceroOpCodes::ACCEPT, 0, outputFile);
        } else if (CAST_MACRO(acceptPartialOp, op,
                              cicero_compiler::dialect::AcceptPartialOp)) {
            writeFunction(CiceroOpCodes::ACCEPT_PARTIAL, 0, outputFile);
        } else if (CAST_MACRO(jumpOp, op, cicero_compiler::dialect::JumpOp)) {
            uint16_t jumpTargetIndex = symbolTable.lookup(jumpOp.getTarget());
            writeFunction(CiceroOpCodes::JUMP, jumpTargetIndex, outputFile);
        } else {
            throw std::runtime_error(
                "Code generation for operation not implemented: " +
                op->getName().getStringRef().str());
        }

        operationIndex++;
    });

    return 0;
}

unique_ptr<RegexParser::AST::Root> getAST() {

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

    return RegexParser::parseRegexFromFile(inputFilename);
}

#define CODEGEN_SEPARATION(opCode, opData)                                     \
    ((opCode & 0x7) << 13) | (opData & 0x1fff)

void outputToFileBinaryFormat(uint16_t opCode, uint16_t opData,
                              ofstream &outputStream) {
    uint16_t toWrite = CODEGEN_SEPARATION(opCode, opData);
    outputStream.write(reinterpret_cast<char *>(&toWrite), sizeof(toWrite));
}
void outputToFileHexFormat(uint16_t opCode, uint16_t opData,
                           ofstream &outputStream) {
    uint16_t toWrite = CODEGEN_SEPARATION(opCode, opData);
    outputStream << "0x" << hex << toWrite << endl;
}

void outputDotFormat(string instructionName,
                     optional<string> instructionArgument,
                     unsigned int instructionIndex,
                     optional<unsigned int> targetIndex, bool linkToNext) {
    cout << instructionIndex << " [label=\"" << instructionName;
    if (instructionArgument.has_value()) {
        cout << ": " << instructionArgument.value();
    }
    cout << "\"];\n";
    if (linkToNext) {
        cout << instructionIndex << " -> " << instructionIndex + 1 << ";\n";
    }
    if (targetIndex.has_value()) {
        cout << instructionIndex << " -> " << targetIndex.value() << ";\n";
    }
}
