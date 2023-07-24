#include "MLIRRegexOptimization.h"
#include <vector>

// #include "mlir-dialect/MyOperationEquivalence.h"
#include "mlir/IR/OperationSupport.h"

using namespace std;

namespace RegexParser::passes {

mlir::LogicalResult optimizeCommonPrefix(mlir::Operation *op,
                                         mlir::PatternRewriter &rewriter);
template <typename OpT>
mlir::LogicalResult checkAllOpInVectorAreEqualAndNotNull(vector<OpT> &ops);

mlir::LogicalResult
FactorizeRoot::matchAndRewrite(RegexParser::dialect::RootOp op,
                               mlir::PatternRewriter &rewriter) const {
    return optimizeCommonPrefix(op.getOperation(), rewriter);
}

mlir::LogicalResult
FactorizeSubregex::matchAndRewrite(RegexParser::dialect::SubRegexOp op,
                                   mlir::PatternRewriter &rewriter) const {
    return optimizeCommonPrefix(op.getOperation(), rewriter);
}

mlir::LogicalResult optimizeCommonPrefix(mlir::Operation *op,
                                         mlir::PatternRewriter &rewriter) {
    mlir::Block &opBlock = op->getRegion(0).front();
    vector<dialect::ConcatenationOp> concatenations;
    concatenations.reserve(10);
    vector<dialect::PieceOp> piecesWalkers;
    piecesWalkers.reserve(10);

    for (auto &op : opBlock) {
        if (auto concat = mlir::dyn_cast<dialect::ConcatenationOp>(op)) {
            auto &maybePieceOp = concat.getBody()->front();
            if (auto pieceOp = mlir::dyn_cast<dialect::PieceOp>(maybePieceOp)) {
                piecesWalkers.emplace_back(std::move(pieceOp));
            } else {
                throw runtime_error(
                    "optimizeCommonPrefix: expected to find PieceOp within "
                    "ConcatenationOp, but found " +
                    maybePieceOp.getName().getStringRef().str());
            }
            concatenations.emplace_back(std::move(concat));
        } else {
            throw runtime_error("optimizeCommonPrefix: block must contain only "
                                "concatenation operations, found " +
                                op.getName().getStringRef().str());
        }
    }

    // We need at least two concatenations to factorize
    if (concatenations.size() < 2) {
        return mlir::failure();
    }

    // Number of identical pieces at the beginning of all the concatenations
    size_t piecesWalked = 0;
    optional<dialect::ConcatenationOp> factorizedConcatenation;
    while (checkAllOpInVectorAreEqualAndNotNull<dialect::PieceOp>(piecesWalkers)
               .succeeded()) {
        // If first iteration, create the factorized concatenation
        if (factorizedConcatenation.has_value() == false) {
            rewriter.setInsertionPointToStart(&opBlock);
            factorizedConcatenation = rewriter.create<dialect::ConcatenationOp>(
                rewriter.getUnknownLoc());
            rewriter.setInsertionPointToStart(
                factorizedConcatenation->getBody());
        }
        // Rewriter insertion point is for sure into the factorized
        // concatenation body, clone one of the pieces (e.g. the first one)
        rewriter.clone(*piecesWalkers[0].getOperation());

        piecesWalked++;
        // Advance all the pieces walkers, while removing them from their
        // respective concatenation
        for (size_t i = 0; i < piecesWalkers.size(); i++) {
            auto nextNode = piecesWalkers[i].getOperation()->getNextNode();
            rewriter.eraseOp(piecesWalkers[i].getOperation());
            if (nextNode == nullptr) {
                piecesWalkers[i] = nullptr;
                continue;
            }
            piecesWalkers[i] = mlir::dyn_cast<dialect::PieceOp>(nextNode);
            if (!piecesWalkers[i]) {
                throw std::runtime_error(
                    "optimizeCommonPrefix: expected to find PieceOp after "
                    "PieceOp, but found " +
                    nextNode->getName().getStringRef().str());
            }
        }
    }

    if (piecesWalked == 0) {
        return mlir::failure();
    }

    /*
     * We have 3 cases now:
     * 1. All the concatenations have been completely factorized, now their
     * bodies are all empty. Just remove them and leave the factorized
     * concatenation. E.G.: A|A => A
     * 2. Some concatenations have been completely factorized, but not all.
     * Remove the ones completely factorized, and append (in an optional
     * subregex) the remaining ones to the factorized concatenation. E.G.:
     * A|Ax|Ay =>  A(x|y)?
     * 3. No concatenation has been completely factorized. Append all the
     * concatenations to the factorized concatenation. E.G. Ax|Ay => A(x|y)
     */

    bool allEmpty = true;
    bool someEmpty = false;
    for (auto &c : concatenations) {
        if (c.getBody()->empty()) {
            someEmpty = true;
        } else {
            allEmpty = false;
        }

        if (someEmpty && !allEmpty)
            break;
    }
    if (allEmpty) {
        // Case 1, just remove the concatenations
        for (auto &c : concatenations) {
            rewriter.eraseOp(c.getOperation());
        }
        return mlir::success();
    }

    // Case 2 and 3

    // The non-empty concatenations have to be moved into a new subregex (for
    // case 2 we attach the optional operation), while being removed from
    // `opBlock`:
    /* Example:
     *
     * factorizedConcatenation {
     *     factorizedPiece1 {[...]}
     *     factorizedPiece2 {[...]}
     *     factorizedPieceN {[...]}
     * }
     * concatenation1 {[...]}
     * concatenation2 { [empty] }
     * concatenationN {[...]}
     *
     * Becomes:
     *
     * factorizedConcatenation {
     *     factorizedPiece1 {[...]}
     *     factorizedPiece2 {[...]}
     *     factorizedPieceN {[...]}
     *     subregexPiece {
     *         subregex {
     *             concatenation1 {[...]}
     *             [concatenation2 is not copied because it is empty]
     *             concatenationN {[...]}
     *         }
     *         quantifier { min=0, min=1 } <- because concatenation2 is empty
     *     }
     * }
     */
    rewriter.setInsertionPointToEnd(factorizedConcatenation->getBody());
    auto subregexPiece =
        rewriter.create<dialect::PieceOp>(rewriter.getUnknownLoc());
    rewriter.setInsertionPointToStart(subregexPiece.getBody());
    auto subregex =
        rewriter.create<dialect::SubRegexOp>(rewriter.getUnknownLoc());
    if (someEmpty /* !!! but not allEmpty !!!*/) {
        // We are in case 2, which means we need to set the subregex as optional
        rewriter.create<dialect::QuantifierOp>(rewriter.getUnknownLoc(), 0, 1);
    }
    rewriter.setInsertionPointToStart(subregex.getBody());
    for (auto &c : concatenations) {
        // For case 2, we only copy non-empty concatenations
        if (!c.getBody()->empty()) {
            rewriter.clone(*c.getOperation());
        }
        rewriter.eraseOp(c.getOperation());
    }

    return mlir::success();
}
template <typename OpT>
mlir::LogicalResult checkAllOpInVectorAreEqualAndNotNull(vector<OpT> &ops) {
    if (ops.size() < 2) {
        return mlir::success();
    }

    if (ops[0] == nullptr) {
        return mlir::failure();
    }

    auto &firstOp = ops[0];
    auto equivalenceFlags = mlir::OperationEquivalence::Flags::IgnoreLocations;
    for (size_t i = 1; i < ops.size(); i++) {

        if (ops[i] == nullptr) {
            return mlir::failure();
        }

        if (false == mlir::OperationEquivalence::isEquivalentTo(
                         ops[i].getOperation(), firstOp.getOperation(),
                         mlir::OperationEquivalence::ignoreValueEquivalence,
                         mlir::OperationEquivalence::ignoreValueEquivalence,
                         equivalenceFlags)) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

} // namespace RegexParser::passes