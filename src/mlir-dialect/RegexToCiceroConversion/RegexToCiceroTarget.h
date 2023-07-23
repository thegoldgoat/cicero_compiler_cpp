#pragma once

#include "CiceroDialectWrapper.h"
#include "RegexDialectWrapper.h"
#include "RegexToCiceroPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace cicero_compiler {

struct RegexToCicero : public mlir::ConversionTarget {
    RegexToCicero(mlir::MLIRContext &ctx) : ConversionTarget(ctx) {
        addLegalOp<mlir::ModuleOp>();
        addLegalDialect<cicero_compiler::dialect::CiceroDialect>();
        addIllegalDialect<RegexParser::dialect::RegexDialect>();
    }
};

mlir::FrozenRewritePatternSet
createRegexToCiceroPatterns(mlir::MLIRContext &ctx) {
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<cicero_compiler::passes::MyConversionPattern>(&ctx);

    return mlir::FrozenRewritePatternSet(std::move(patterns));
}

} // namespace cicero_compiler