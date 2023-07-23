#include "RegexToCiceroPasses.h"

namespace cicero_compiler::passes {
mlir::LogicalResult
MyConversionPattern::matchAndRewrite(RegexParser::dialect::MatchAnyCharOp op,
                                     mlir::PatternRewriter &rewriter) const {
    return mlir::LogicalResult::failure();
}
}