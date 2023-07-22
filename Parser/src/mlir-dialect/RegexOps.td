#ifndef CICERO_REGEX_DIALECT_H
#define CICERO_REGEX_DIALECT_H

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Dialect definition
def Regex_Dialect : Dialect {
  let name = "regex";
  let cppNamespace = "::RegexParser::dialect";
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

// Base operation definition
class Regex_Op<string mnemonic, list<Trait> traits = []> : Op<Regex_Dialect, mnemonic, traits> {}

def AlternationOp : Regex_Op<"alternation"> {
    let summary = "Alternation of Concatenations";
    let description = [{
        Match if any of the concatenation matches
    }];
}

def ConcatenationOp : Regex_Op<"concatenation"> {
    let summary = "Concatenations of Pieces";
    let description = [{
        Match if all the pieces match in sequence
    }];
}

def PieceOp : Regex_Op<"piece"> {
    let summary = "Atom with optional quantifier";
    let description = [{
        Match if we find <atom> repeated <quantifier> times
    }];
}

def QuantifierOp : Regex_Op<"quantifier"> {
    let summary = "Quantifier that specifies the repetition of a certain atom";
    let description = [{
        Match if we find <atom> repeated <quantifier> times
    }];

    let arguments = (ins I64Attr:$min, I64Attr:$max);
}

// class AtomOp<string mnemonic> : Regex_Op<mnemonic> {
//     let summary = "Abstract Atom";
// }

def MatchCharOp : Regex_Op<"match_char"> {
    let summary = "Match a single character";
    let description = [{
        Match if the character is equal to the one specified
    }];

    let arguments = (ins I8Attr:$targetChar);
}

def MatchAnyCharOp : Regex_Op<"match_any_char"> {
    let summary = "Match any character";
    let description = [{
        Match any character
    }];
}

def GroupOp : Regex_Op<"group"> {
    let summary = "Group of character";
    let description = [{
        Match if any of the characters in the group is found
    }];
}


def SubRegexOp : Regex_Op<"sub_regex"> {
    let summary = "Sub-Regex";
    let description = [{
        Match if the sub-regex matches
    }];
}

#endif // CICERO_REGEX_DIALECT_H