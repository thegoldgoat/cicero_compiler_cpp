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

def SplitOp : Cicero_Op<"split", [Symbol, SymbolTable]> {
    let summary = "Splits the thread of execution into two";
    let description = [{
        The thread is split into two threads:
        - one continues at the next operation
        - the other one continues at the **split body** (its entry block)
    }];

    let regions = (region AnyRegion:$splittedThread);

    let builders = [
        OpBuilder<(ins), [{
            // build($_builder, $_state);
            $_state.addRegion();
            $_state.regions[0]->push_back(new Block());
        }]>
    ];

    let extraClassDeclaration = [{
        mlir::Block *getBody() {
            return &getSplittedThread().getBlocks().front();
        }

        bool isOptionalSymbol() { return true; }
    }];

    let hasCanonicalizer = 1;
}

def FlatSplitOp : Cicero_Op<"flat_split"> {
    let summary = "Splits the thread of execution into two";
    let description = [{
        The thread is split into two threads:
        - one continues at the next operation
        - the other one continues at operation referenced by the splitTarget attribute
    }];

    let arguments = (ins FlatSymbolRefAttr:$splitTarget);
}

def JumpOp : Cicero_Op<"jump", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, Terminator]> {
    let summary = "Unconditionally jumps to argument";
    let description = [{
        The thread continues at the instruction specified by the argument as a symbol reference.
    }];

    let arguments = (ins FlatSymbolRefAttr:$target);
}

def MatchAnyOp : Cicero_Op<"match_any"> {
    let summary = "Matches any character";
    let description = [{
        Matches any character, advances the input by one character.
    }];
}

def MatchCharOp : Cicero_Op<"match_char"> {
    let summary = "Matches a specific character";
    let description = [{
        Matches a specific character given as argument, advances the input by one character.
    }];

    let arguments = (ins I8Attr:$targetChar);
}

def NotMatchCharOp : Cicero_Op<"not_match_char"> {
    let summary = "Matches any character except the one given as argument";
    let description = [{
        Matches any character except the one given as argument, advances the input by one character.
        Complementary to match_char.
    }];

    let arguments = (ins I8Attr:$targetChar);
}

def PlaceholderOp : Cicero_Op<"placeholder", [Symbol]> {
    let summary = "Placeholder operation";
    let description = [{
        Placeholder operation, only used internally by the compiler to represent jump/split targets.
    }];
}

#endif // CICERO_DIALECT_H