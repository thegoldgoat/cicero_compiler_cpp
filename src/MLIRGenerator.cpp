#include "MLIRGenerator.h"

using namespace std;

mlir::ModuleOp cicero_compiler::MLIRGenerator::mlirGen(
    unique_ptr<RegexParser::AST::RegExp> regExp, bool isRoot) {
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

    populateRegexBody(module.getBody(), *regExp);

    builder.create<cicero_compiler::dialect::AcceptOp>(builder.getUnknownLoc());
    return module;
}

void cicero_compiler::MLIRGenerator::populateRegexBody(
    mlir::Block *block, const RegexParser::AST::RegExp &regExp) {
    auto &concatenations = regExp.getConcatenations();

    if (concatenations.size() == 0) {
        throw std::runtime_error("Regex body was empty. This should have been "
                                 "caught earlier (e.g. in the parser)");
    }

    if (concatenations.size() == 1) {
        populateConcatenateBody(block, *concatenations[0]);
        return;
    }

    auto endSymbol = getNewSymbolName();

    /*
     * [concat1, concat2, concat3]
     *
     * SPLIT {
     *     <concat1>
     *     JUMP(endSymbol)
     * }
     * SPLIT {
     *     <concat2>
     *     JUMP(endSymbol)
     * }
     * <concat3>
     * endSymbol: isRoot ? accept : placeholder
     */
    for (size_t i = 0; i < concatenations.size() - 1; i++) {
        auto &concatenation = concatenations[i];
        builder.setInsertionPointToEnd(block);

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

    populateConcatenateBody(block, *concatenations.back());

    auto end = builder.create<cicero_compiler::dialect::PlaceholderOp>(
        builder.getUnknownLoc());
    end.setName(endSymbol);
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

    auto singleChar = dynamic_cast<const RegexParser::AST::Char *>(&atom);
    if (singleChar != nullptr) {
        builder.create<cicero_compiler::dialect::MatchCharOp>(
            builder.getUnknownLoc(), singleChar->getChar());
        return;
    }

    auto anyChar = dynamic_cast<const RegexParser::AST::AnyChar *>(&atom);
    if (anyChar != nullptr) {
        builder.create<cicero_compiler::dialect::MatchAnyOp>(
            builder.getUnknownLoc());
        return;
    }

    auto subRegex = dynamic_cast<const RegexParser::AST::SubRegex *>(&atom);
    if (subRegex != nullptr) {
        populateRegexBody(block, subRegex->getRegExp());
        return;
    }

    auto group = dynamic_cast<const RegexParser::AST::Group *>(&atom);
    if (group != nullptr) {
        pupulateGroupBody(block, *group);
        return;
    }
}

void cicero_compiler::MLIRGenerator::pupulateGroupBody(
    mlir::Block *block, const RegexParser::AST::Group &group) {
    /*
     * In total, we can potentially match charsToMatch.size() characters.
     * If we need to match more than half of those, it's more efficient to
     * not_match the rest of the characters. Otherwise, we can match the
     * list of characters.
     * Examples:
     * [abc]
     * Split {match(a) jump(endSymbol)}
     * Split {match(b) jump(endSymbol)}
     * match(c)
     * endSymbol: placeholder
     *
     * [^abc]
     * Split {not_match(a) jump(endSymbol)}
     * Split {not_match(b) jump(endSymbol)}
     * not_match(c)
     * endSymbol: placeholder
     */
    auto &charsToMatch = group.getCharsToMatch();

    // Count of true values in charsToMatch
    vector<bool>::size_type countToMatch =
        std::count(charsToMatch.begin(), charsToMatch.end(), true);

    bool flippedGroup = (countToMatch > charsToMatch.size() / 2);

    auto endSymbol = getNewSymbolName();

    auto matchCreatorNotFlipped = [this](char c) {
        builder.create<cicero_compiler::dialect::MatchCharOp>(
            builder.getUnknownLoc(), c);
    };

    auto matchCreatorFlipped = [this](char c) {
        builder.create<cicero_compiler::dialect::NotMatchCharOp>(
            builder.getUnknownLoc(), c);
    };

    auto matchCreator = flippedGroup ? std::function(matchCreatorFlipped)
                                     : std::function(matchCreatorNotFlipped);

    int lastIndex = -1;
    for (vector<bool>::size_type i = 0; i != charsToMatch.size(); i++) {
        /*
            +-----------------+--------------+---------------------+
            | charsToMatch[i] | flippedGroup | action              |
            +-----------------+--------------+---------------------+
            | true            | true         | None                |
            +-----------------+--------------+---------------------+
            | true            | false        | split(matchChar)    |
            +-----------------+--------------+---------------------+
            | false           | true         | split(notMatchChar) |
            +-----------------+--------------+---------------------+
            | false           | false        | None                |
            +-----------------+--------------+---------------------+
        */
        if (charsToMatch[i] != flippedGroup) {
            if (lastIndex != -1) {
                auto split = builder.create<cicero_compiler::dialect::SplitOp>(
                    builder.getUnknownLoc());
                auto splitBody = new mlir::Block();
                split.getSplittedThread().push_back(splitBody);

                builder.setInsertionPointToStart(splitBody);
                matchCreator(lastIndex);
                builder.create<cicero_compiler::dialect::JumpOp>(
                    builder.getUnknownLoc(), endSymbol);

                builder.setInsertionPointAfter(split);
            }
            lastIndex = i;
        }
    }

    if (lastIndex == -1) {
        // Group was empty (or full)? Cannot match anything
        throw std::runtime_error("Group was empty (or full)? Cannot match "
                                 "anything. This should have been caught "
                                 "earlier (e.g. in the parser)");
    }

    // Match (or not match) the last char
    matchCreator(lastIndex);

    builder
        .create<cicero_compiler::dialect::PlaceholderOp>(
            builder.getUnknownLoc())
        .setName(endSymbol);

    return;
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
