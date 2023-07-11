#pragma once
#include "AST.h"
#include "DialectWrapper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>

namespace cicero_compiler {
class MLIRGenerator {
  public:
    MLIRGenerator(mlir::MLIRContext &context) : builder(&context){};

    mlir::ModuleOp mlirGen(std::unique_ptr<RegexParser::AST::RegExp> regExp,
                           bool isRoot = true);

  private:
    mlir::OpBuilder builder;
    mlir::ModuleOp module;

    unsigned int symbolCounter = 0;
    std::string getNewSymbolName();
};

} // namespace cicero_compiler