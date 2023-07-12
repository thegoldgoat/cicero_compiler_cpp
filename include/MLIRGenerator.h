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

    void populateRegexBody(mlir::Block *block,
                           const RegexParser::AST::RegExp &regExp);

    void populateConcatenateBody(
        mlir::Block *block,
        const RegexParser::AST::Concatenation &concatenation);

    void populateAtomBody(mlir::Block *block,
                          const RegexParser::AST::Atom &atom);

    void pupulateGroupBody(mlir::Block *block,
                           const RegexParser::AST::Group &group);

    void populateQuantifierOptionalBody(mlir::Block *block,
                              const RegexParser::AST::Atom &atom);

    void populateQuantifierStarBody(mlir::Block *block,
                              const RegexParser::AST::Atom &atom);
    
    void populateQuantifierPlusBody(mlir::Block *block, 
                              const RegexParser::AST::Atom &atom);

    void populateQuantifierRangeBody(mlir::Block *block,
                              const RegexParser::AST::Atom &atom, int min, int max);

  private:
    mlir::OpBuilder builder;
    mlir::ModuleOp module;

    unsigned int symbolCounter = 0;
    std::string getNewSymbolName();
};

} // namespace cicero_compiler