#pragma once

#include "CiceroDialectWrapper.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace cicero_compiler::passes {

using namespace cicero_compiler::dialect;

/// @brief Simplify the jump operations based on target
/// @details This pass updates the jump operation these cases:
/// 1. Jump to the next instruction (the jump is just removed)
/// 2. Jump to a jump (JMP(JMP(x)) -> JMP(x))
/// 3. Jump to an accept (JMP(ACC) -> ACC)
/// 4. Jump to an accept_partial (JMP(ACC_PAR) -> ACC_PAR)
struct SimplifyJump : public mlir::OpRewritePattern<JumpOp> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyJump)

    SimplifyJump(mlir::MLIRContext *context)
        : OpRewritePattern(context, /*benefit=*/10) {}

    mlir::LogicalResult
    matchAndRewrite(JumpOp op, mlir::PatternRewriter &rewriter) const override;
};

/// @brief Flatten the split operations
/// @details This pass updates the split operation by removing the content of
/// the split body, appending it to the end of the block, and replacing the
/// split operation with a flat_split operation.
struct FlattenSplit : public mlir::OpRewritePattern<SplitOp> {

    FlattenSplit(mlir::MLIRContext *context)
        : OpRewritePattern(context, /*benefit=*/1000) {}

    mlir::LogicalResult
    matchAndRewrite(SplitOp op, mlir::PatternRewriter &rewriter) const override;
};

/// @brief Removes all placeholder operations
struct PlaceholderRemover : public mlir::OpRewritePattern<PlaceholderOp> {
    PlaceholderRemover(mlir::MLIRContext *context)
        : OpRewritePattern(context, /*benefit=*/100) {}

    mlir::LogicalResult
    matchAndRewrite(PlaceholderOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

/// @brief Removes this operation, and moves its symbol to the next operation
/// @details This pass updates the operation by removing it, if it has a symbol
/// then it moves the symbol to the next operation. If the next operation
/// already has a symbol, then it will be replaced and all the uses of the
/// replaced symbol will be corrected accordingly.
/// @param op operation to remove
/// @param rewriter pattern rewriter class to use
mlir::LogicalResult removeOperationAndMoveSymbolToNext(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter);

} // namespace cicero_compiler::passes
