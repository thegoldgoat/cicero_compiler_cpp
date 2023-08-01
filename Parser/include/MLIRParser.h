#pragma once

#include "MLIRRegexOptimization.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <string>

namespace RegexParser {

/// @brief Parse a regex from a file
/// @param context MLIR context to use
/// @param regexPath Path to the file that contains the regex
/// @param printFileToStdout if true, the content of the file is printed to
/// stdout before parsing
/// @return the module containing the regex dialect operations
mlir::ModuleOp parseRegexFromFile(mlir::MLIRContext &context,
                                  const std::string &regexPath,
                                  bool printFileToStdout = true);
/// @brief Parse a regex from an in-memory string
/// @param context MLIR context to use
/// @param regex string containing the regex
/// @return the module containing the regex dialect operations
mlir::ModuleOp parseRegexFromString(mlir::MLIRContext &context,
                                    const std::string &regex);

/// @brief Run optimization passes on the regex module
/// @param module the module to optimize
/// @return success if the optimization passes are successful, failure otherwise
mlir::LogicalResult optimizeRegex(mlir::ModuleOp &module);

} // namespace RegexParser
