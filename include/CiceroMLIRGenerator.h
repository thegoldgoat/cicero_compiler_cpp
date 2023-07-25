#pragma once

#include "CiceroDialectWrapper.h"
#include "RegexDialectWrapper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace cicero_compiler {
class CiceroMLIRGenerator {
  public:
    CiceroMLIRGenerator(mlir::MLIRContext &context) : builder(&context){};

    mlir::ModuleOp mlirGen(mlir::ModuleOp &regexModule);

  private:
    mlir::ModuleOp mlirGen(RegexParser::dialect::RootOp &regexRoot);

    void populateConcatenationFather(mlir::Block *block,
                                     mlir::Operation *alternationFather);

    void populateConcatenation(mlir::Block *block,
                               RegexParser::dialect::ConcatenationOp &op);

    void populatePiece(mlir::Block *block, RegexParser::dialect::PieceOp &op);

    void populateAtom(mlir::Block *block, mlir::Operation &atom);

    void populateQuantifier(mlir::Block *block,
                            RegexParser::dialect::QuantifierOp &op,
                            mlir::Operation &atom);

    void populateGroup(mlir::Block *block, RegexParser::dialect::GroupOp &op);

    mlir::OpBuilder builder;

    unsigned int symbolCounter = 0;
    std::string getNewSymbolName();
};

} // namespace cicero_compiler