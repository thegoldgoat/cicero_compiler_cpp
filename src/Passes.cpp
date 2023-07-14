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