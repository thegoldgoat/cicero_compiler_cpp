#include "Passes.h"
#include "CiceroDialectWrapper.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include <iostream>

namespace cicero_compiler::passes {
using namespace cicero_compiler::dialect;

unsigned int symbolCounter = 0;

mlir::LogicalResult
FlattenSplit::matchAndRewrite(SplitOp op,
                              mlir::PatternRewriter &rewriter) const {

    std::string splitTarget = "FLATTEN_" + std::to_string(symbolCounter++);

    rewriter.setInsertionPointToEnd(op.getOperation()->getBlock());
    rewriter.create<PlaceholderOp>(op.getLoc(), splitTarget);

    rewriter.mergeBlocks(op.getBody(), op.getOperation()->getBlock());
    rewriter.create<JumpOp>(op.getLoc(), op.getSplitReturnAttr());

    rewriter.setInsertionPointAfter(op.getOperation());
    auto flatsplitOp = rewriter.replaceOpWithNewOp<FlatSplitOp>(
        op.getOperation(), splitTarget);
    flatsplitOp.setName(op.getNameAttr());

    return mlir::success();
}

mlir::LogicalResult
PlaceholderRemover::matchAndRewrite(PlaceholderOp op,
                                    mlir::PatternRewriter &rewriter) const {
    return removeOperationAndMoveSymbolToNext(op.getOperation(), rewriter);
}

mlir::LogicalResult
SimplifyJump::matchAndRewrite(JumpOp op,
                              mlir::PatternRewriter &rewriter) const {
    auto targetOp = mlir::SymbolTable::lookupNearestSymbolFrom(
        op.getOperation(), op.getTargetAttr());

    if (!targetOp) {
        op.emitError("Jump operation has invalid target?!?");
        throw std::runtime_error("Jump operation has invalid target?!?");
    }

    if (targetOp == op.getOperation()->getNextNode()) {
        return removeOperationAndMoveSymbolToNext(op.getOperation(), rewriter);
    }

    if (auto otherOpJump = mlir::dyn_cast<JumpOp>(targetOp)) {
        op.setTargetAttr(otherOpJump.getTargetAttr());
        return mlir::success();
    }

    if (auto otherOpAccept = mlir::dyn_cast<AcceptOp>(targetOp)) {
        rewriter.replaceOpWithNewOp<AcceptOp>(op.getOperation());
        return mlir::success();
    }

    if (auto otherOpAccept = mlir::dyn_cast<AcceptPartialOp>(targetOp)) {
        rewriter.replaceOpWithNewOp<AcceptPartialOp>(op.getOperation());
        return mlir::success();
    }

    return mlir::failure();
}

mlir::LogicalResult
removeOperationAndMoveSymbolToNext(mlir::Operation *op,
                                   mlir::PatternRewriter &rewriter) {
    auto opSymbol = mlir::SymbolTable::getSymbolName(op);

    auto nextOp = op->getNextNode();

    if (!nextOp) {
        op->emitWarning("Trying to move symbol to the next operation, but "
                        "there is no next operation, failing "
                        "gracefully. My operation name is: '" +
                        op->getName().getStringRef().str() + "'; " +
                        "My symbol is: '" +
                        mlir::SymbolTable::getSymbolName(op).str() + "'");
        return mlir::failure();
    }

    auto nextOpSymbol = mlir::SymbolTable::getSymbolName(nextOp);

    // The next operation does not have a symbol, simply assign my symbol
    if (!nextOpSymbol) {
        mlir::SymbolTable::setSymbolName(nextOp, opSymbol);
        rewriter.eraseOp(op);
        return mlir::success();
    }

    // Otherwise, replace all uses of the operation symbol with the symbol of
    // the next one
    for (auto walker = op; walker; walker = walker->getParentOp()) {
        if (mlir::SymbolTable::replaceAllSymbolUses(op, nextOpSymbol, walker)
                .failed()) {
            throw std::runtime_error(
                "During PlaceholderRemover, failed to replace all symbol "
                "uses of the placeholder symbol with the next operation's "
                "symbol");
        }
    }

    rewriter.eraseOp(op);
    return mlir::success();
}

} // namespace cicero_compiler::passes
