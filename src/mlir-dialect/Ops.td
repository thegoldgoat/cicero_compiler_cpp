#ifndef CICERO_DIALECT_H
#define CICERO_DIALECT_H

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


// Dialect definition
def Cicero_Dialect : Dialect {
  let name = "cicero";
  let cppNamespace = "::cicero_compiler::dialect";
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

// Base operation definition
class Cicero_Op<string mnemonic, list<Trait> traits = []> : Op<Cicero_Dialect, mnemonic, traits> {
}


// *** Operation definitions


def AcceptOp : Cicero_Op<"accept"> {
    let summary = "Accepts the input only at the end of string";
    let description = [{
        If we are at the end of the string, then the input is accepted, otherwise thread is killed.
    }];
}

def AcceptPartialOp : Cicero_Op<"accept_partial"> {
    let summary = "Always accepts the input";
    let description = [{
        Input gets accepted, regardless of the current position in the string.
    }];
}

#endif // CICERO_DIALECT_H