#pragma once

#include "CiceroDialectWrapper.h"
#include "RegexDialectWrapper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace cicero_compiler {

/// @brief Helper class to generate a MLIR cicero module from a MLIR regex
/// module
class CiceroMLIRGenerator {
  public:
    CiceroMLIRGenerator(mlir::MLIRContext &context) : builder(&context){};

    /// @brief Generate a MLIR cicero module from a MLIR regex module
    /// @param regexModule input MLIR regex module
    /// @return generated MLIR cicero module
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

  private:
    /// @brief Handle the case when we have a group where it is more convenient
    /// to match the positive case: for example [abc], we may want to create
    /// three threads that match a, b and c
    void populatePositiveGroup(mlir::Block *block,
                               RegexParser::dialect::GroupOp &op);
    /// @brief  Handle the case when we have a group where it is more convenient
    /// to match the negative case: for example [^abc], we may want to just
    /// not_match(a) -> not_match(b) -> not_match(c) -> match_any
    void populateNegativeGroup(mlir::Block *block,
                               RegexParser::dialect::GroupOp &op);
};

} // namespace cicero_compiler