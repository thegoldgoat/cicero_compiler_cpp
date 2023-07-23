#pragma once

#include "CiceroDialectWrapper.h"
#include "RegexDialectWrapper.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace cicero_compiler::passes {

struct MyConversionPattern
    : public mlir::OpRewritePattern<RegexParser::dialect::MatchAnyCharOp> {
    MyConversionPattern(mlir::MLIRContext *context)
        : OpRewritePattern(context) {}

    mlir::LogicalResult
    matchAndRewrite(RegexParser::dialect::MatchAnyCharOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

} // namespace cicero_compiler::passes