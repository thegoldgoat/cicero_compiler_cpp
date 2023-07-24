#include "MLIRRegexOptimization.h"

using namespace std;

namespace RegexParser::passes {

/// @brief Get the first operation within father body, check it is a
/// concatenation and that it has no next operation
/// @param father The operation whose body contains the concatenation
/// @return A ConcatenationOp, if we found a single concatenation within father
/// body
template <typename OpT>
dialect::ConcatenationOp getFirstAndOnlyConcatenation(OpT &father) {
    auto firstConcatenation =
        mlir::dyn_cast<dialect::ConcatenationOp>(father.getBody()->front());

    if (!firstConcatenation) {
        father.emitError("getFirstAndOnlyConcatenation: expected to find a "
                         "ConcatenationOp within father, here is the father: ")
            << father;
        return nullptr;
    }

    if (firstConcatenation.getOperation()->getNextNode()) {
        // More than one operation found
        return nullptr;
    }

    return firstConcatenation;
}

mlir::LogicalResult SimplifySubregexNotQuantified::matchAndRewrite(
    RegexParser::dialect::SubRegexOp op,
    mlir::PatternRewriter &rewriter) const {
    auto firstConcatenation =
        getFirstAndOnlyConcatenation<dialect::SubRegexOp>(op);

    if (!firstConcatenation) {
        return mlir::failure();
    }

    if (op.getOperation()->getNextNode()) {
        // There is an operation after subregex (which should be quantifier):
        // cannot optimize
        return mlir::failure();
    }

    auto fatherPiece = op.getOperation()->getParentOp();
    rewriter.setInsertionPointAfter(fatherPiece);

    // Clone all the pieces within the subregex and place them after the father
    // of subregex
    for (auto &pieceInSubregex :
         firstConcatenation.getBody()->getOperations()) {
        rewriter.clone(pieceInSubregex);
    }

    // Remove father of subregex (and also subregex)
    rewriter.eraseOp(fatherPiece);
    return mlir::success();
}

mlir::LogicalResult SimplifySubregexSinglePiece::matchAndRewrite(
    dialect::SubRegexOp op, mlir::PatternRewriter &rewriter) const {
    auto firstConcatenation =
        getFirstAndOnlyConcatenation<dialect::SubRegexOp>(op);

    if (!firstConcatenation) {
        return mlir::failure();
    }

    auto firstPiece =
        mlir::dyn_cast<dialect::PieceOp>(firstConcatenation.getBody()->front());

    if (!firstPiece) {
        op.emitError(
            "SimplifySubregexSinglePiece: expected to find a "
            "PieceOp within concatenation, here is the ConcatenationOp: ")
            << firstConcatenation;
        throw runtime_error("SimplifySubregexSinglePiece: expected to find a "
                            "PieceOp within concatenation.");
    }

    if (firstPiece.getOperation()->getNextNode()) {
        // This subregex contains more than one piece, cannot optimize
        return mlir::failure();
    }

    // Ok, we are sure we have only one concatenation and one piece, get the
    // QuantifierOp of the SubRegexOp and the QuantifierOp of this `firstPiece`:
    dialect::QuantifierOp quantifierRegex = nullptr, quantifierPiece = nullptr;

    auto nodeAfterSubregex = op.getOperation()->getNextNode();
    if (nodeAfterSubregex) {
        quantifierRegex =
            mlir::dyn_cast<dialect::QuantifierOp>(nodeAfterSubregex);

        if (!quantifierRegex) {
            op.emitError("SimplifySubregexSinglePiece: expected to find a "
                         "QuantifierOp after SubRegexOp, here is the operation "
                         "after SubRegexOp: ")
                << nodeAfterSubregex;
            throw runtime_error(
                "SimplifySubregexSinglePiece: expected to find a "
                "QuantifierOp after SubRegexOp.");
        }
    }

    mlir::Operation &atomInPiece = firstPiece.getBody()->front();

    if (atomInPiece.getNextNode()) {
        quantifierPiece =
            mlir::dyn_cast<dialect::QuantifierOp>(atomInPiece.getNextNode());
        if (!quantifierPiece) {
            op.emitError("SimplifySubregexSinglePiece: expected to find a "
                         "QuantifierOp after PieceOp, here is the operation "
                         "after PieceOp: ")
                << atomInPiece.getNextNode();
            throw runtime_error("SimplifySubregexSinglePiece: expected to find "
                                "a QuantifierOp after PieceOp.");
        }
    }

    // Cannot optimize if we have both quantifiers
    if (quantifierRegex && quantifierPiece) {
        return mlir::failure();
    }

    // Move the atom in the father of the subregex
    rewriter.setInsertionPointAfter(op.getOperation());
    rewriter.clone(atomInPiece);

    // If we only had quantifierRegex, it is already in place. If we only had
    // quantifierPiece, we need to clone it after the atom we just moved.
    if (quantifierPiece) {
        rewriter.clone(*quantifierPiece.getOperation());
    }

    // Remove the subregex and return success
    rewriter.eraseOp(op);
    return mlir::success();
}

} // namespace RegexParser::passes