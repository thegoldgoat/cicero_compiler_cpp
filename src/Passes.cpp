#include "DialectWrapper.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include <iostream>

using namespace cicero_compiler::dialect;


struct RemovePlaceholders : public mlir::OpRewritePattern<PlaceholderOp> {

    RemovePlaceholders(mlir::MLIRContext *context)
        : OpRewritePattern(context, /*benefit=*/100) {}

    mlir::LogicalResult
    matchAndRewrite(PlaceholderOp op,
                    mlir::PatternRewriter &rewriter) const override {
        return mlir::LogicalResult::success();
    }
};

void cicero_compiler::dialect::PlaceholderOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
    results.add<RemovePlaceholders>(context);
}