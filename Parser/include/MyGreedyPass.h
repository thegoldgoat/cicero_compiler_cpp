#pragma once

#include "mlir/Pass/Pass.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

/// @brief Simple pass implementation that just runs a set of patterns greedily
/// @tparam OpT type of the root operation
template <typename OpT>
struct MyGreedyPass
    : public mlir::PassWrapper<MyGreedyPass<OpT>, mlir::OperationPass<OpT>> {
    /// @param patterns set of patterns to apply greedily
    /// @param config configuration to use for the greedy rewrite driver
    MyGreedyPass(mlir::FrozenRewritePatternSet &&patterns,
                 mlir::GreedyRewriteConfig config)
        : frozenPatterns(patterns), config(config) {}

    void runOnOperation() override {
        if (mlir::applyPatternsAndFoldGreedily(this->getOperation(),
                                               frozenPatterns, config)
                .failed()) {
            this->signalPassFailure();
        }
    }

  private:
    mlir::FrozenRewritePatternSet frozenPatterns;
    mlir::GreedyRewriteConfig config;
};

/// @brief Run a set of rewrite patterns greedily
/// @tparam OpT type of the root operation
/// @param op root operation
/// @param patterns set of rewrite patterns to apply
/// @param config configuration to use for the greedy rewrite driver
/// @return success if the patterns are successfully applied, failure otherwise
template <typename OpT>
mlir::LogicalResult runMyGreedyPass(mlir::Operation *op,
                                    mlir::FrozenRewritePatternSet &&patterns,
                                    mlir::GreedyRewriteConfig config) {
    mlir::PassManager pm(op->getContext());
    pm.addPass(std::make_unique<MyGreedyPass<OpT>>(
        mlir::FrozenRewritePatternSet(std::move(patterns)),
        mlir::GreedyRewriteConfig()));

    return pm.run(op);
}