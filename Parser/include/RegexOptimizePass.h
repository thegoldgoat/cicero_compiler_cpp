#include "mlir/Pass/Pass.h"

#include "RegexDialectWrapper.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace RegexParser::passes {

struct RegexOptimizePass
    : public mlir::PassWrapper<
          RegexOptimizePass, mlir::OperationPass<RegexParser::dialect::RootOp>> {
    RegexOptimizePass(mlir::MLIRContext *context) {
        mlir::RewritePatternSet patterns(context);
        patterns.add<passes::FactorizeRoot, passes::FactorizeSubregex,
                     passes::SimplifySubregexNotQuantified,
                     passes::SimplifySubregexSinglePiece,
                     passes::SimplifyLeadingQuantifiers>(context);

        frozenPatterns = std::move(patterns);
    }

    void runOnOperation() override {
        if (mlir::applyPatternsAndFoldGreedily(getOperation()->getParentOp(), frozenPatterns,
                                               config)
                .failed()) {
            signalPassFailure();
        }
    }

  private:
    mlir::FrozenRewritePatternSet frozenPatterns;
    mlir::GreedyRewriteConfig config;
};

std::unique_ptr<mlir::Pass> createRegexOptimizePass(mlir::MLIRContext *context) {
    return std::make_unique<RegexOptimizePass>(context);
}

} // namespace RegexParser::passes
