#pragma once

#include "RegexDialectWrapper.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace RegexParser::passes {

#define DEFINE_REWRITE_PATTERN_MACRO(patternName, opName)                            \
    struct patternName : public mlir::OpRewritePattern<opName> {               \
        patternName(mlir::MLIRContext *context) : OpRewritePattern(context) {} \
        mlir::LogicalResult                                                    \
        matchAndRewrite(opName op,                                             \
                        mlir::PatternRewriter &rewriter) const override;       \
    }

/*
 * Factorization 
*/

DEFINE_REWRITE_PATTERN_MACRO(FactorizeRoot, RegexParser::dialect::RootOp);
DEFINE_REWRITE_PATTERN_MACRO(FactorizeSubregex, RegexParser::dialect::SubRegexOp);

} // namespace RegexParser::passes