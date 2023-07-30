#include "CiceroPasses.h"
#include "CiceroDialectWrapper.h"
#include "SplitFollowers.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include <iostream>

namespace cicero_compiler::passes {
using namespace cicero_compiler::dialect;

unsigned int symbolCounter = 0;

mlir::LogicalResult
FlattenSplit::matchAndRewrite(SplitOp op,
                              mlir::PatternRewriter &rewriter) const {
    bool needToAddJump = true;

    // If the split body ends with an accept (e.g. it comes from a concatenation
    // that ended with $) then we do not need to add the jump, as the accept is
    // already a `terminator`
    if (op.getBody()->empty() == false &&
        mlir::dyn_cast<dialect::AcceptOp>(op.getBody()->back())) {
        needToAddJump = false;
    }

    // If the split body is empty (e.g. a?b) then we can just set the
    // `splitTarget` of `FlattenSplitOp` to the `splitReturn` of SplitOp
    if (op.getBody()->empty()) {
        auto flatsplitOp = rewriter.replaceOpWithNewOp<FlatSplitOp>(
            op.getOperation(), op.getSplitReturnAttr());
        flatsplitOp.setName(op.getNameAttr());
        return mlir::success();
    }

    std::string splitTarget = "F" + std::to_string(symbolCounter++);

    rewriter.setInsertionPointAfter(op);

    for (auto &op : op.getBody()->getOperations()) {
        rewriter.clone(op);
    }

    if (needToAddJump) {
        rewriter.create<JumpOp>(op.getLoc(), op.getSplitReturnAttr());
    }
    rewriter.create<PlaceholderOp>(op.getLoc(), splitTarget);

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
        op.emitError("Jump operation has invalid target: ")
            << op.getTargetAttr();
        op.getOperation()->getParentOfType<mlir::ModuleOp>().dump();
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
        auto replaced =
            rewriter.replaceOpWithNewOp<AcceptOp>(op.getOperation());
        replaced.setName(op.getNameAttr());
        return mlir::success();
    }

    if (auto otherOpAccept = mlir::dyn_cast<AcceptPartialOp>(targetOp)) {
        auto replaced =
            rewriter.replaceOpWithNewOp<AcceptPartialOp>(op.getOperation());
        replaced.setName(op.getNameAttr());
        return mlir::success();
    }

    return mlir::failure();
}

mlir::LogicalResult
removeOperationAndMoveSymbolToNext(mlir::Operation *op,
                                   mlir::PatternRewriter &rewriter) {
    auto opSymbol = mlir::SymbolTable::getSymbolName(op);

    // If it does not have a symbol, just remove it
    if (!opSymbol) {
        rewriter.eraseOp(op);
        return mlir::success();
    }

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

#include <iostream>

mlir::LogicalResult
SplitMerger::matchAndRewrite(FlatSplitOp op,
                             mlir::PatternRewriter &rewriter) const {
    auto splitFollowers = SplitFollowers();

    splitFollowers.optimizeByFactorize(op, rewriter);

    return mlir::failure();
}

} // namespace cicero_compiler::passes
