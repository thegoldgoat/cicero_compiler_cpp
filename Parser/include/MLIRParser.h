#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <string>

namespace RegexParser {

mlir::ModuleOp parseRegexFromFile(mlir::MLIRContext &context,
                                  const std::string &regexPath);
mlir::ModuleOp parseRegexFromString(mlir::MLIRContext &context,
                                    const std::string &regex);

} // namespace RegexParser
