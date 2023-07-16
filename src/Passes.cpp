#include "DialectWrapper.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include <iostream>

using namespace cicero_compiler::dialect;

unsigned int symbolCounter = 0;

struct FlattenSplit : public mlir::OpRewritePattern<SplitOp> {

    FlattenSplit(mlir::MLIRContext *context)
        : OpRewritePattern(context, /*benefit=*/100) {}

    mlir::LogicalResult
    matchAndRewrite(SplitOp op,
                    mlir::PatternRewriter &rewriter) const override {

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
};

void cicero_compiler::dialect::SplitOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
    results.add<FlattenSplit>(context);
}

struct PlaceholderRemover : public mlir::OpRewritePattern<PlaceholderOp> {
    PlaceholderRemover(mlir::MLIRContext *context)
        : OpRewritePattern(context, /*benefit=*/1000) {}

    mlir::LogicalResult
    matchAndRewrite(PlaceholderOp op,
                    mlir::PatternRewriter &rewriter) const override {

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
};

void cicero_compiler::dialect::PlaceholderOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
    results.add<PlaceholderRemover>(context);
}