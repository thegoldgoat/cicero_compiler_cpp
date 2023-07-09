#include "DialectWrapper.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <iostream>

int main() {
    mlir::MLIRContext context;
    context.getOrLoadDialect<cicero_compiler::dialect::CiceroDialect>();

    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    builder.setInsertionPointToStart(module.getBody());

    builder.create<cicero_compiler::dialect::AcceptOp>(builder.getUnknownLoc());
    builder.create<cicero_compiler::dialect::AcceptPartialOp>(builder.getUnknownLoc());

    module.dump();
    return 0;
}