#include "MLIRRegexOptimization.h"
#include <vector>

// #include "mlir-dialect/MyOperationEquivalence.h"
#include "mlir/IR/OperationSupport.h"

using namespace std;

namespace RegexParser::passes {

mlir::LogicalResult optimizeCommonPrefix(mlir::Operation *op,
                                         mlir::PatternRewriter &rewriter);
template <typename OpT>
mlir::LogicalResult checkAllOpInVectorAreEqual(vector<OpT> &ops);

mlir::LogicalResult
FactorizeRoot::matchAndRewrite(RegexParser::dialect::RootOp op,
                               mlir::PatternRewriter &rewriter) const {
    return optimizeCommonPrefix(op.getOperation(), rewriter);
}

mlir::LogicalResult
FactorizeSubregex::matchAndRewrite(RegexParser::dialect::SubRegexOp op,
                                   mlir::PatternRewriter &rewriter) const {
    op == op;
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

    unsigned int piecesWalked = 0;
    while (true) {
        // Check all pieces are the same
        if (checkAllOpInVectorAreEqual<dialect::PieceOp>(piecesWalkers)
                .succeeded()) {
            piecesWalked++;
            for (size_t i = 0; i < piecesWalkers.size(); i++) {
                auto nextNode = piecesWalkers[i].getOperation()->getNextNode();
                if (nextNode == nullptr) {
                    // TODO: Check that also all the other ones are at the end
                    // of the block

                    // We reached the end of the block! Remove all the
                    // concatenations except the first one
                    for (size_t j = 1; j < concatenations.size(); j++) {
                        rewriter.eraseOp(concatenations[j].getOperation());
                    }
                    return mlir::success();
                }
                auto nextPiece = mlir::dyn_cast<dialect::PieceOp>(nextNode);
                if (nextPiece) {
                    piecesWalkers[i] = nextPiece;
                } else {
                    throw std::runtime_error(
                        "optimizeCommonPrefix: expected to find PieceOp after "
                        "PieceOp, but found " +
                        piecesWalkers[i]
                            .getOperation()
                            ->getNextNode()
                            ->getName()
                            .getStringRef()
                            .str());
                }
            }
        } else {
            break;
        }
    }

    if (piecesWalked == 0) {
        return mlir::failure();
    }

    // Get the first `piecesWalked` pieces from the any concatenation (e.g. the
    // first) and put them in a new ConcatenationOp, at the beginning of
    // `opBlock`. Then, remove the first `piecesWalked` pieces from all the
    // other concatenations.

    rewriter.setInsertionPointToStart(&opBlock);
    auto newConcatenation =
        rewriter.create<dialect::ConcatenationOp>(rewriter.getUnknownLoc());
    auto newConcatenationBody = newConcatenation.getBody();
    rewriter.setInsertionPointToStart(newConcatenationBody);

    size_t piecesAlreadyMoved = 0;
    for (auto &op : concatenations[0].getBody()->getOperations()) {
        rewriter.clone(op);
        piecesAlreadyMoved++;
        if (piecesAlreadyMoved == piecesWalked) {
            break;
        }
    }

    for (auto &concatenation : concatenations) {
        for (size_t i = 0; i < piecesWalked; i++) {
            rewriter.eraseOp(&concatenation.getBody()->front());
        }
    }

    return mlir::success();
}
template <typename OpT>
mlir::LogicalResult checkAllOpInVectorAreEqual(vector<OpT> &ops) {
    assert(ops.size() > 0);
    if (ops.size() == 1) {
        return mlir::success();
    }

    auto &firstOp = ops[0];
    auto equivalenceFlags = mlir::OperationEquivalence::Flags::IgnoreLocations;
    for (size_t i = 1; i < ops.size(); i++) {
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