#include "cicero_helper.h"
#include "CiceroDialectWrapper.h"
#include "cicero_const.h"
#include "cicero_macros.h"

#include <iostream>

using namespace std;

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

void outputDotFormat(const string &nodeLabel, const string &nodeColor,
                     unsigned int instructionIndex,
                     optional<unsigned int> targetIndex, bool linkToNext) {
    cout << instructionIndex << " [label=\"" << nodeLabel
         << "\" color=\"black\" style=\"filled\" fillcolor=\"" << nodeColor
         << "\"]\n";
    if (linkToNext) {
        cout << instructionIndex << " -> " << instructionIndex + 1 << ";\n";
    }
    if (targetIndex.has_value()) {
        cout << instructionIndex << " -> " << targetIndex.value() << ";\n";
    }
}

namespace cicero_compiler {

void dumpCiceroDot(mlir::ModuleOp &module) {
    auto symbolTable = createSymbolTable(module);
    unsigned int operationIndex = 0;

    cout << "digraph {" << endl
         << "begin [label=\"begin\"];" << endl
         << "begin -> 0;" << endl;
    module.getBody()->walk([&symbolTable,
                            &operationIndex](mlir::Operation *op) {
        bool isLastInstruction = op->getNextNode() == nullptr;
        if (CAST_MACRO(matchCharOp, op,
                       cicero_compiler::dialect::MatchCharOp)) {
            outputDotFormat(string(1, matchCharOp.getTargetChar()),
                            CICERO_COLOR_MATCH, operationIndex, nullopt,
                            !isLastInstruction);
        } else if (CAST_MACRO(notMatchOp, op,
                              cicero_compiler::dialect::NotMatchCharOp)) {
            outputDotFormat("Not " + string(1, notMatchOp.getTargetChar()),
                            CICERO_COLOR_MATCH, operationIndex, nullopt,
                            !isLastInstruction);
        } else if (CAST_MACRO(matchAnyOp, op,
                              cicero_compiler::dialect::MatchAnyOp)) {
            outputDotFormat(".", CICERO_COLOR_MATCH, operationIndex, nullopt,
                            !isLastInstruction);
        } else if (CAST_MACRO(flatSplitOp, op,
                              cicero_compiler::dialect::FlatSplitOp)) {
            uint16_t splitTargetIndex =
                symbolTable[flatSplitOp.getSplitTarget().str()];

            if (isLastInstruction) {
                throw std::runtime_error(
                    "Last instruction of program cannot be a split, how "
                    "did we get here???");
            }

            outputDotFormat("Split", CICERO_COLOR_SPLIT, operationIndex,
                            splitTargetIndex, true);

        } else if (CAST_MACRO(acceptOp, op,
                              cicero_compiler::dialect::AcceptOp)) {
            outputDotFormat("Accept", CICERO_COLOR_ACCEPT, operationIndex,
                            nullopt, false);
        } else if (CAST_MACRO(acceptPartialOp, op,
                              cicero_compiler::dialect::AcceptPartialOp)) {
            outputDotFormat("AcceptPartial", CICERO_COLOR_ACCEPT,
                            operationIndex, nullopt, false);
        } else if (CAST_MACRO(jumpOp, op, cicero_compiler::dialect::JumpOp)) {
            uint16_t jumpTargetIndex = symbolTable[jumpOp.getTarget().str()];

            outputDotFormat("Jump", CICERO_COLOR_JUMP, operationIndex,
                            jumpTargetIndex, false);
        } else {
            throw std::runtime_error(
                "Graphviz output for operation not implemented: " +
                op->getName().getStringRef().str());
        }
        operationIndex++;
    });

    cout << "}" << endl;
}

void dumpCompiled(mlir::ModuleOp &module, std::ofstream &outputFile,
                  CiceroBinaryOutputFormat format) {
    auto symbolTable = cicero_compiler::createSymbolTable(module);

    unsigned int operationIndex = 0;
    auto writeFunction =
        format == Hex ? outputToFileHexFormat : outputToFileBinaryFormat;
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
                symbolTable[flatSplitOp.getSplitTarget().str()];

            writeFunction(CiceroOpCodes::SPLIT, splitTargetIndex, outputFile);
        } else if (CAST_MACRO(acceptOp, op,
                              cicero_compiler::dialect::AcceptOp)) {
            writeFunction(CiceroOpCodes::ACCEPT, 0, outputFile);
        } else if (CAST_MACRO(acceptPartialOp, op,
                              cicero_compiler::dialect::AcceptPartialOp)) {
            writeFunction(CiceroOpCodes::ACCEPT_PARTIAL, 0, outputFile);
        } else if (CAST_MACRO(jumpOp, op, cicero_compiler::dialect::JumpOp)) {
            uint16_t jumpTargetIndex = symbolTable[jumpOp.getTarget().str()];

            writeFunction(CiceroOpCodes::JUMP, jumpTargetIndex, outputFile);
        } else {
            throw std::runtime_error(
                "Code generation for operation not implemented: " +
                op->getName().getStringRef().str());
        }

        operationIndex++;
    });
}

std::unordered_map<std::string, unsigned int>
createSymbolTable(mlir::ModuleOp &module) {
    std::unordered_map<std::string, unsigned int> symbolTable;
    unsigned int operationIndex = 0;
    module.getBody()->walk(
        [&symbolTable, &operationIndex](mlir::Operation *op) {
            auto opSymbol = mlir::SymbolTable::getSymbolName(op);
            if (opSymbol) {
                symbolTable[opSymbol.getValue().str()] = operationIndex;
            }
            operationIndex++;
        });

    return std::move(symbolTable);
}

} // namespace cicero_compiler