#include "MLIRGenerator.h"

using namespace std;

mlir::ModuleOp cicero_compiler::MLIRGenerator::mlirGen(
    unique_ptr<RegexParser::AST::RegExp> regExp, bool isRoot) {
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

    auto &concatenations = regExp->getConcatenations();

    auto endSymbol = getNewSymbolName();

    // For each concatenation, create a SplitOp that has as body:
    // 1. Concatenation body
    // 2. JumpOp to the end of this module
    for (auto &concatenation : concatenations) {
        builder.setInsertionPointToEnd(module.getBody());

        auto concatenationSplit =
            builder.create<cicero_compiler::dialect::SplitOp>(
                builder.getUnknownLoc());

        // Create a new block for the concatenation body
        auto splitBody = new mlir::Block();
        concatenationSplit.getSplittedThread().push_back(splitBody);
        builder.setInsertionPointToStart(splitBody);

        // TODO: Replace this fake body with a real one
        for (auto &symbol : concatenation->getSymbols()) {
            builder.create<cicero_compiler::dialect::MatchAnyOp>(
                builder.getUnknownLoc());
        }

        builder.create<cicero_compiler::dialect::JumpOp>(
            builder.getUnknownLoc(), endSymbol);
    }

    builder.setInsertionPointToEnd(module.getBody());

    if (isRoot) {
        auto end = builder.create<cicero_compiler::dialect::AcceptOp>(
            builder.getUnknownLoc());

        cout << "endSymbol: " << endSymbol.data() << endl;
        end.setName(endSymbol);
        ((char *)endSymbol.data())[0] = 'A';
        cout << "endSymbol after update: " << endSymbol.data() << endl;
        cout << "getName after update: " << end.getName().data() << endl;

    } else {
        auto end = builder.create<cicero_compiler::dialect::PlaceholderOp>(
            builder.getUnknownLoc());
        end.setName(endSymbol);
    }
    return module;
}

std::string cicero_compiler::MLIRGenerator::getNewSymbolName() {
    return "S" + std::to_string(symbolCounter++);
}
