#pragma once

#define CAST_MACRO(resultName, inputOperation, operationType)                  \
    auto resultName = mlir::dyn_cast<operationType>(inputOperation)

