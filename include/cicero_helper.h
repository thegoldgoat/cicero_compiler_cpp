#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "cicero_const.h"

#include <unordered_map>
#include <fstream>
#include <memory>

namespace cicero_compiler {

void dumpCompiled(mlir::ModuleOp &module, std::ofstream &outputFile, CiceroBinaryOutputFormat format);

void dumpCiceroDot(mlir::ModuleOp &module);

std::unordered_map<std::string, unsigned int>
createSymbolTable(mlir::ModuleOp &module);

}