#include "Passes.h"
#include "DialectWrapper.h"
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

    auto nextOp = op.getOperation()->getNextNode();

    if (!nextOp) {
        op.emitError("PlaceholderOp must be followed by some other "
                     "operation??? Or should we just remove it and assign "
                     "my symbol to operation before me?");
        return mlir::failure();
    }

    auto nextOpSymbol = nextOp->getAttrOfType<mlir::StringAttr>(
        mlir::SymbolTable::getSymbolAttrName());

    if (!nextOpSymbol) {
        mlir::SymbolTable::setSymbolName(nextOp, op.getName());
        rewriter.eraseOp(op);
        return mlir::success();
    }

    for (auto walker = op.getOperation(); walker;
         walker = walker->getParentOp()) {

        if (mlir::SymbolTable::replaceAllSymbolUses(op.getOperation(),
                                                    nextOpSymbol, walker)
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
        rewriter.eraseOp(op);
        return mlir::success();
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

} // namespace cicero_compiler::passes
