#pragma once

#include "DialectWrapper.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace cicero_compiler::passes {

using namespace cicero_compiler::dialect;
struct SimplifyJump : public mlir::OpRewritePattern<JumpOp> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyJump)

    SimplifyJump(mlir::MLIRContext *context)
        : OpRewritePattern(context, /*benefit=*/100) {}

    mlir::LogicalResult
    matchAndRewrite(JumpOp op, mlir::PatternRewriter &rewriter) const override;
};

struct FlattenSplit : public mlir::OpRewritePattern<SplitOp> {

    FlattenSplit(mlir::MLIRContext *context)
        : OpRewritePattern(context, /*benefit=*/100) {}

    mlir::LogicalResult
    matchAndRewrite(SplitOp op, mlir::PatternRewriter &rewriter) const override;
};

struct PlaceholderRemover : public mlir::OpRewritePattern<PlaceholderOp> {
    PlaceholderRemover(mlir::MLIRContext *context)
        : OpRewritePattern(context, /*benefit=*/1000) {}

    mlir::LogicalResult
    matchAndRewrite(PlaceholderOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

} // namespace cicero_compiler::passes
