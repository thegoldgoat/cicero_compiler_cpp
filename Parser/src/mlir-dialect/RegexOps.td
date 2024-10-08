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
class Regex_Op<string mnemonic, list<Trait> traits = []> : Op<Regex_Dialect, mnemonic, !listconcat(traits, [NoTerminator])> {}

def RootOp : Regex_Op<"root", [IsolatedFromAbove]> {
    let summary = "Root of the Regex";
    let description = [{
        Match if the regex matches
    }];

    let regions = (region AnyRegion:$rootRegion);
    let arguments = (ins BoolAttr:$hasPrefix, BoolAttr:$hasSuffix);

    let builders = [
        OpBuilder<(ins "bool":$hasPrefix, "bool":$hasSuffix), [{
            $_state.addRegion();
            $_state.regions[0]->push_back(new Block());

            $_state.addAttribute(getHasPrefixAttrName($_state.name), ::mlir::BoolAttr::get($_builder.getContext(), hasPrefix));
            $_state.addAttribute(getHasSuffixAttrName($_state.name), ::mlir::BoolAttr::get($_builder.getContext(), hasSuffix));
        }]>
    ];

    let extraClassDeclaration = [{
        mlir::Block *getBody() {
            return &getRootRegion().getBlocks().front();
        }
    }];

    let assemblyFormat = [{
        attr-dict regions
    }];
}

def ConcatenationOp : Regex_Op<"concatenation"> {
    let summary = "Concatenations of Pieces";
    let description = [{
        Match if all the pieces match in sequence
    }];

    let regions = (region AnyRegion:$concatenationRegion);

    let builders = [
        OpBuilder<(ins ), [{
            $_state.addRegion();
            $_state.regions[0]->push_back(new Block());
        }]>
    ];

    let extraClassDeclaration = [{
        mlir::Block *getBody() {
            return &getConcatenationRegion().getBlocks().front();
        }
    }];

    let assemblyFormat = [{
        attr-dict regions
    }];
}

def PieceOp : Regex_Op<"piece"> {
    let summary = "Atom with optional quantifier";
    let description = [{
        Match if we find <atom> repeated <quantifier> times
    }];

    let regions = (region AnyRegion:$pieceRegion);

    let builders = [
        OpBuilder<(ins ), [{
            $_state.addRegion();
            $_state.regions[0]->push_back(new Block());
        }]>
    ];

    let extraClassDeclaration = [{
        mlir::Block *getBody() {
            return &getPieceRegion().getBlocks().front();
        }
    }];

    let assemblyFormat = [{
        attr-dict regions
    }];
}

def QuantifierOp : Regex_Op<"quantifier"> {
    let summary = "Quantifier that specifies the repetition of a certain atom";
    let description = [{
        Match if we find <atom> repeated <quantifier> times
    }];

    let arguments = (ins SI64Attr:$min, SI64Attr:$max);

    let assemblyFormat = [{
        attr-dict `from` $min `to` $max
    }];
}

def MatchCharOp : Regex_Op<"match_char"> {
    let summary = "Match a single character";
    let description = [{
        Match if the character is equal to the one specified
    }];

    let arguments = (ins I8Attr:$targetChar);

    let hasCustomAssemblyFormat = 1;
}

def NotMatchCharOp : Regex_Op<"not_match_char"> {
    let summary = "Match anything but single character";
    let description = [{
        Match if the character is not equal to the one specified
    }];

    let arguments = (ins I8Attr:$targetChar);

    let hasCustomAssemblyFormat = 1;
}

def MatchAnyCharOp : Regex_Op<"match_any_char"> {
    let summary = "Match any character";
    let description = [{
        Match any character
    }];

    let assemblyFormat = [{
        attr-dict
    }];
}

def GroupOp : Regex_Op<"group"> {
    let summary = "Group of character";
    let description = [{
        Match if any of the characters in the group is found
    }];

    let arguments = (ins DenseBoolArrayAttr:$targetChars);

    let assemblyFormat = [{
        attr-dict $targetChars
    }];
}


def SubRegexOp : Regex_Op<"sub_regex"> {
    let summary = "Sub-Regex";
    let description = [{
        Match if the sub-regex matches
    }];

    let regions = (region AnyRegion:$subregexRegion);

    let builders = [
        OpBuilder<(ins ), [{
            $_state.addRegion();
            $_state.regions[0]->push_back(new Block());
        }]>
    ];

    let extraClassDeclaration = [{
        mlir::Block *getBody() {
            return &getSubregexRegion().getBlocks().front();
        }
    }];

    let assemblyFormat = [{
        attr-dict regions
    }];
}

def DollarOp : Regex_Op<"dollar"> {
    let summary = "Match the end of the string";
    let description = [{
        Match if the end of the string is reached, can only be added as the last piece of a concatenation, with no quantifiers.
    }];

    let assemblyFormat = [{
        attr-dict
    }];
}

#endif // CICERO_REGEX_DIALECT_H