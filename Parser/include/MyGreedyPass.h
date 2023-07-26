#pragma once

#include "mlir/Pass/Pass.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

template <typename OpT>
struct MyGreedyPass
    : public mlir::PassWrapper<MyGreedyPass<OpT>, mlir::OperationPass<OpT>> {
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

template <typename OpT>
mlir::LogicalResult runMyGreedyPass(mlir::MLIRContext *context,
                                    mlir::Operation *op,
                                    mlir::FrozenRewritePatternSet &&patterns,
                                    mlir::GreedyRewriteConfig config) {
    mlir::PassManager pm(context);
    pm.addPass(std::make_unique<MyGreedyPass<OpT>>(
        mlir::FrozenRewritePatternSet(std::move(patterns)),
        mlir::GreedyRewriteConfig()));

    return pm.run(op);
}