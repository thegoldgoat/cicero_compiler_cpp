#pragma once

#include "cicero_const.h"
#include "mlir/IR/BuiltinOps.h"

#include <fstream>
#include <memory>
#include <unordered_map>

namespace cicero_compiler {

/// @brief Dump the compiled artifact of the module to an output file
/// @param module The module whose artifact we want to dump
/// @param outputFile The output file stream to dump the artifact to
/// @param format The format of the output file
void dumpCompiled(mlir::ModuleOp &module, std::ofstream &outputFile,
                  CiceroBinaryOutputFormat format);

/// @brief Dump the cicero MLIR operations in a graphviz format
/// @param module The module whose operations we want to dump
void dumpCiceroDot(mlir::ModuleOp &module);

} // namespace cicero_compiler