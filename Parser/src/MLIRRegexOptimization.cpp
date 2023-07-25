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

/// @brief Get the first operation that has siblings while climbing up the
/// operation tree
/// @tparam LimitOpType type of operation that, when encountered, means we have
/// to stop (we stop at the child of LimitOpType)
/// @param op the first operation that needs to be checked for siblings. Cannot
/// be of type LimitOpType
/// @return the first operation that has siblings while climbing up the
/// operation tree
template <typename LimitOpType>
mlir::Operation *getFirstOpWithSiblings(mlir::Operation *op) {
    // Check I have no siblings
    if (!op->getNextNode() && !op->getPrevNode()) {
        auto parent = op->getParentOp();
        // Stop at LimitOpType child
        if (mlir::dyn_cast<LimitOpType>(parent)) {
            return op;
        }
        return getFirstOpWithSiblings<LimitOpType>(parent);
    }

    return op;
}

// TODO: Rewrite taking into consideration modified ConcatenationOp
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
                rewriter.getUnknownLoc(), false, false);
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

mlir::LogicalResult SimplifyLeadingQuantifiers::matchAndRewrite(
    dialect::RootOp op, mlir::PatternRewriter &rewriter) const {
    // Iterate all concatenations
    bool oneModification = false;
    vector<mlir::Operation *> operationToRemove;
    operationToRemove.reserve(4);
    for (auto &concat : op.getBody()->getOperations()) {
        auto concatenationOp = mlir::dyn_cast<dialect::ConcatenationOp>(concat);
        if (!concatenationOp) {
            op.emitError("SimplifyLeadingQuantifiers: expected to find "
                         "ConcatenationOp within RootOp, but found ")
                << concat;
            throw std::runtime_error(
                "SimplifyLeadingQuantifiers: expected to find ConcatenationOp "
                "within RootOp");
        }

        // Get last piece
        auto &lastPiece = concatenationOp.getBody()->back();
        auto pieceOp = mlir::dyn_cast<dialect::PieceOp>(lastPiece);
        if (!pieceOp) {
            op.emitError("SimplifyLeadingQuantifiers: expected to find "
                         "PieceOp within ConcatenationOp, but found ")
                << lastPiece;
            throw std::runtime_error(
                "SimplifyLeadingQuantifiers: expected to find PieceOp within "
                "ConcatenationOp");
        }

        // Check it has quantifier
        auto &quantifier = pieceOp.getBody()->getOperations().back();
        auto quantifierOp = mlir::dyn_cast<dialect::QuantifierOp>(quantifier);
        // If we do not have a quantifier, we cannot optimize.
        if (!quantifierOp) {
            continue;
        }
        uint64_t min = quantifierOp.getMin();
        uint64_t max = quantifierOp.getMax();
        // If min is already equal to max, means we cannot optimize any further
        if (min == max) {
            continue;
        }

        oneModification = true;

        // If min == 0, then directly remove PieceOp
        if (min == 0) {
            operationToRemove.push_back(getFirstOpWithSiblings<dialect::RootOp>(
                pieceOp.getOperation()));
            continue;
        }

        quantifierOp.setMax(min);
    }

    if (oneModification) {
        for (auto op : operationToRemove) {
            rewriter.eraseOp(op);
        }
        return mlir::success();
    }
    return mlir::failure();
}

} // namespace RegexParser::passes