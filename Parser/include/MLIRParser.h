#pragma once

#include "MLIRRegexOptimization.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <string>

namespace RegexParser {

mlir::ModuleOp parseRegexFromFile(mlir::MLIRContext &context,
                                  const std::string &regexPath,
                                  bool printFileToStdout = true);
mlir::ModuleOp parseRegexFromString(mlir::MLIRContext &context,
                                    const std::string &regex);

mlir::LogicalResult optimizeRegex(mlir::ModuleOp &module);

} // namespace RegexParser
