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

        populateConcatenateBody(splitBody, *concatenation);

        builder.create<cicero_compiler::dialect::JumpOp>(
            builder.getUnknownLoc(), endSymbol);
    }

    builder.setInsertionPointToEnd(module.getBody());

    if (isRoot) {
        auto end = builder.create<cicero_compiler::dialect::AcceptOp>(
            builder.getUnknownLoc());

        end.setName(endSymbol);
    } else {
        auto end = builder.create<cicero_compiler::dialect::PlaceholderOp>(
            builder.getUnknownLoc());
        end.setName(endSymbol);
    }
    return module;
}

void cicero_compiler::MLIRGenerator::populateConcatenateBody(
    mlir::Block *block, const RegexParser::AST::Concatenation &concatenation) {
    builder.setInsertionPointToStart(block);

    for (auto &piece : concatenation.getPieces()) {
        auto &atom = piece->getAtom();
        auto &quantifier = piece->getQuantifier();

        if (piece->hasQuantifier()) {
            switch (quantifier.getType()) {
            case RegexParser::AST::QuantifierType::OPTIONAL:
                populateQuantifierOptionalBody(block, atom);
                break;
            case RegexParser::AST::QuantifierType::STAR:
                populateQuantifierStarBody(block, atom);
                break;
            case RegexParser::AST::QuantifierType::PLUS:
                populateQuantifierPlusBody(block, atom);
                break;
            case RegexParser::AST::QuantifierType::RANGE:
                populateQuantifierRangeBody(block, atom, quantifier.getMin(),
                                            quantifier.getMax());
                break;
            default:
                throw std::runtime_error(
                    "Quantifier type not implemented yet: " +
                    quantifier.getType());
            }
        } else {
            populateAtomBody(block, atom);
        }
    }
}

void cicero_compiler::MLIRGenerator::populateAtomBody(
    mlir::Block *block, const RegexParser::AST::Atom &atom) {
    builder.setInsertionPointToEnd(block);

    // TODO: Replace this fake body with a real one
    builder.create<cicero_compiler::dialect::MatchAnyOp>(
        builder.getUnknownLoc());
}

void cicero_compiler::MLIRGenerator::populateQuantifierOptionalBody(
    mlir::Block *block, const RegexParser::AST::Atom &atom) {
    /*
        SPLIT(END) {
            <atomBody>
            Jump(endSymbol)
        }
        endSymbol:  placeholder
        -> builderCursor
     */
    auto endSymbol = getNewSymbolName();
    auto split = builder.create<cicero_compiler::dialect::SplitOp>(
        builder.getUnknownLoc());
    auto splitBody = new mlir::Block();
    split.getSplittedThread().push_back(splitBody);
    populateAtomBody(splitBody, atom);
    builder.create<cicero_compiler::dialect::JumpOp>(builder.getUnknownLoc(),
                                                     endSymbol);
    builder.setInsertionPointAfter(split);
    auto placeholder = builder.create<cicero_compiler::dialect::PlaceholderOp>(
        builder.getUnknownLoc());
    placeholder.setName(endSymbol);
}

void cicero_compiler::MLIRGenerator::populateQuantifierStarBody(
    mlir::Block *block, const RegexParser::AST::Atom &atom) {
    /*
        splitSymbol: SPLIT(END) {
            <atomBody>
            JUMP(splitSymbol)
        }
        -> builderCursor
    */
    auto splitSymbol = getNewSymbolName();

    auto split = builder.create<cicero_compiler::dialect::SplitOp>(
        builder.getUnknownLoc());
    split.setName(splitSymbol);

    auto splitBody = new mlir::Block();
    split.getSplittedThread().push_back(splitBody);
    populateAtomBody(splitBody, atom);

    builder.create<cicero_compiler::dialect::JumpOp>(builder.getUnknownLoc(),
                                                     splitSymbol);

    builder.setInsertionPointAfter(split);
}

void cicero_compiler::MLIRGenerator::populateQuantifierPlusBody(
    mlir::Block *block, const RegexParser::AST::Atom &atom) {
    /*
        <atomBody>
        splitSymbol: SPLIT(END) {
            <atomBody>
            JUMP(splitSymbol)
        }
        -> builderCursor
    */
    populateAtomBody(block, atom);
    populateQuantifierStarBody(block, atom);
}

void cicero_compiler::MLIRGenerator::populateQuantifierRangeBody(
    mlir::Block *block, const RegexParser::AST::Atom &atom, int min, int max) {
    if (max == -1) {
        // upper bound is infinity
        for (int i = 0; i < min; i++) {
            populateAtomBody(block, atom);
        }
        populateQuantifierStarBody(block, atom);
        return;
    }

    if (min == max) {
        // Repeat exactly `min` times
        for (int i = 0; i < min; i++) {
            populateAtomBody(block, atom);
        }
        return;
    }

    // In general, repeat `min` times, then optionally repeat `max - min` times
    for (int i = 0; i < min; i++) {
        populateAtomBody(block, atom);
    }

    for (int i = 0; i < max - min; i++) {
        populateQuantifierOptionalBody(block, atom);
    }
}

std::string cicero_compiler::MLIRGenerator::getNewSymbolName() {
    return "S" + std::to_string(symbolCounter++);
}
