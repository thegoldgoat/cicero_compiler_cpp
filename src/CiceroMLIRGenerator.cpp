#include "CiceroMLIRGenerator.h"

using namespace std;

namespace cicero_compiler {

using namespace RegexParser::dialect;

mlir::ModuleOp CiceroMLIRGenerator::mlirGen(mlir::ModuleOp &regexModule) {
    auto rootOp = mlir::dyn_cast<RootOp>(&regexModule.getBody()->front());

    return mlirGen(rootOp);
}

mlir::ModuleOp
CiceroMLIRGenerator::mlirGen(RegexParser::dialect::RootOp &regexRoot) {
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

    if (regexRoot.getHasPrefix()) {
        builder.setInsertionPointToStart(module.getBody());
        auto prefixSplitSymbol = std::string("PREFIX_SPLIT");

        auto prefixSplitOp = builder.create<cicero_compiler::dialect::SplitOp>(
            builder.getUnknownLoc(), prefixSplitSymbol);

        builder.setInsertionPointToStart(prefixSplitOp.getBody());

        prefixSplitOp.setName(prefixSplitSymbol);

        builder.create<cicero_compiler::dialect::MatchAnyOp>(
            builder.getUnknownLoc());
    }

    populateConcatenationFather(module.getBody(), regexRoot.getOperation());

    if (regexRoot.getHasSuffix()) {
        builder.create<cicero_compiler::dialect::AcceptPartialOp>(
            builder.getUnknownLoc());
    } else {
        builder.create<cicero_compiler::dialect::AcceptOp>(
            builder.getUnknownLoc());
    }

    return module;
}

void CiceroMLIRGenerator::populateConcatenationFather(
    mlir::Block *block, mlir::Operation *alternationFather) {

    auto &concatenations =
        alternationFather->getRegion(0).getBlocks().front().getOperations();

    if (concatenations.empty()) {
        alternationFather->emitError(
            "Concatenation father does not contain any operation?");
        throw std::runtime_error("");
    }

    // If we have only one concatenation, just populate the concatenation
    if (concatenations.front().getNextNode() == nullptr) {
        if (auto concatenationOp =
                mlir::dyn_cast<ConcatenationOp>(&concatenations.front())) {
            populateConcatenation(block, concatenationOp);
        } else {
            throw std::runtime_error(
                "Expected ConcatenationOp during populateConcatenationFather, "
                "instead found " +
                concatenations.front().getName().getStringRef().str());
        }
        return;
    }

    auto endSymbol = getNewSymbolName();

    for (auto &op : concatenations) {
        if (auto concatenationOp = mlir::dyn_cast<ConcatenationOp>(&op)) {

            if (op.getNextNode() == nullptr) {
                // This is the last concatenation, so no need for a split
                populateConcatenation(block, concatenationOp);
                break;
            }

            builder.setInsertionPointToEnd(block);

            auto concatenationSplit =
                builder.create<cicero_compiler::dialect::SplitOp>(
                    builder.getUnknownLoc(), endSymbol);

            populateConcatenation(concatenationSplit.getBody(),
                                  concatenationOp);

        } else {
            throw std::runtime_error(
                "Expected ConcatenationOp during populateConcatenationFather, "
                "instead found " +
                op.getName().getStringRef().str());
        }
    }

    builder.create<cicero_compiler::dialect::PlaceholderOp>(
        builder.getUnknownLoc(), endSymbol);
}

void CiceroMLIRGenerator::populateConcatenation(
    mlir::Block *block, RegexParser::dialect::ConcatenationOp &op) {

    for (auto &op : op.getBody()->getOperations()) {
        if (auto pieceOp = mlir::dyn_cast<PieceOp>(&op)) {
            builder.setInsertionPointToEnd(block);
            populatePiece(block, pieceOp);
        } else {
            throw std::runtime_error(
                "Expected PieceOp during populateConcatenation, "
                "instead found " +
                op.getName().getStringRef().str());
        }
    }
}

void CiceroMLIRGenerator::populatePiece(mlir::Block *block,
                                        RegexParser::dialect::PieceOp &op) {
    auto &childrenOps = op.getBody()->getOperations();
    auto &atomOp = childrenOps.front();

    auto quantifierOperation = atomOp.getNextNode();

    if (!quantifierOperation) {
        // No quantifier, just populate the atom
        populateAtom(block, atomOp);
    } else {
        if (auto quantifierOp =
                mlir::dyn_cast<QuantifierOp>(*quantifierOperation)) {
            populateQuantifier(block, quantifierOp, atomOp);
        } else {
            throw std::runtime_error(
                "Expected QuantifierOp during populatePiece as second "
                "Operation, "
                "instead found " +
                quantifierOperation->getName().getStringRef().str());
        }
    }
}

void CiceroMLIRGenerator::populateAtom(mlir::Block *block,
                                       mlir::Operation &atom) {
    builder.setInsertionPointToEnd(block);

    if (auto matchCharOp = mlir::dyn_cast<MatchCharOp>(&atom)) {
        builder.create<cicero_compiler::dialect::MatchCharOp>(
            builder.getUnknownLoc(), matchCharOp.getTargetCharAttr());
        return;
    }

    if (auto matchAnyCharOp = mlir::dyn_cast<MatchAnyCharOp>(&atom)) {
        builder.create<cicero_compiler::dialect::MatchAnyOp>(
            builder.getUnknownLoc());
        return;
    }

    if (auto subregexOp = mlir::dyn_cast<SubRegexOp>(&atom)) {
        populateConcatenationFather(block, &atom);
        return;
    }

    if (auto groupOp = mlir::dyn_cast<GroupOp>(&atom)) {
        populateGroup(block, groupOp);
        return;
    }

    throw std::runtime_error("Invalid atom type, found" +
                             atom.getName().getStringRef().str());
}

void CiceroMLIRGenerator::populateQuantifier(
    mlir::Block *block, RegexParser::dialect::QuantifierOp &op,
    mlir::Operation &atom) {
    auto min = op.getMin();
    auto max = op.getMax();

    // Add `min` times the atom
    for (auto count = min; count > 0; count--) {
        populateAtom(block, atom);
    }

    if (max == -1) {
        // Repeat possibly infinite times
        auto splitSymbol = getNewSymbolName();

        builder.setInsertionPointToEnd(block);

        auto splitOp = builder.create<cicero_compiler::dialect::SplitOp>(
            builder.getUnknownLoc(), splitSymbol);
        splitOp.setName(splitSymbol);

        builder.setInsertionPointToStart(splitOp.getBody());

        populateAtom(splitOp.getBody(), atom);

    } else {
        auto endSymbol = getNewSymbolName();
        for (; max > min; max--) {
            builder.setInsertionPointToEnd(block);
            auto splitOp = builder.create<cicero_compiler::dialect::SplitOp>(
                builder.getUnknownLoc(), endSymbol);

            builder.setInsertionPointToStart(splitOp.getBody());

            populateAtom(block, atom);
        }
        builder.create<cicero_compiler::dialect::PlaceholderOp>(
            builder.getUnknownLoc(), endSymbol);
    }

    builder.setInsertionPointToEnd(block);
}

void CiceroMLIRGenerator::populateGroup(mlir::Block *block,
                                        RegexParser::dialect::GroupOp &op) {
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
    auto charsToMatch = op.getTargetChars();

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
    for (size_t i = 0; i != charsToMatch.size(); i++) {
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
                    builder.getUnknownLoc(), endSymbol);

                builder.setInsertionPointToStart(split.getBody());
                matchCreator(lastIndex);

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

    builder.create<cicero_compiler::dialect::PlaceholderOp>(
        builder.getUnknownLoc(), endSymbol);
}

std::string CiceroMLIRGenerator::getNewSymbolName() {
    return "S" + std::to_string(symbolCounter++);
}
} // namespace cicero_compiler