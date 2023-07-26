#pragma once

#include "RegexDialectWrapper.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace RegexParser::passes {

#define DEFINE_REWRITE_PATTERN_MACRO(patternName, opName)                      \
    struct patternName : public mlir::OpRewritePattern<opName> {               \
        patternName(mlir::MLIRContext *context) : OpRewritePattern(context) {} \
        mlir::LogicalResult                                                    \
        matchAndRewrite(opName op,                                             \
                        mlir::PatternRewriter &rewriter) const override;       \
    }

/*
 * Factorization: Factorize root/subregex which alternates between concatenation
 * with same prefix, for example:
 * 1. `this|that` -> `th(is|at)`
 * 2. `this|that|th` -> `th(is|at)?`
 * 3. `^this|^that` -> `^th(is|at)`
 * 4. `this|^that` -> `this|^that`
 * 5. `ab|ab$` -> `ab` <- note that only root concatenation can have `$`, hence
 * for those it can be simplified
 * 6. `ab|ab$|abc` -> `ab` <- same as above
 * 7. `ab|ab` -> `ab`
 */

DEFINE_REWRITE_PATTERN_MACRO(FactorizeRoot, RegexParser::dialect::RootOp);
DEFINE_REWRITE_PATTERN_MACRO(FactorizeSubregex,
                             RegexParser::dialect::SubRegexOp);

/*
 * Simplify a subregex which contains a single concatenation and it is not
 quantified:
    [...]
    Piece {
        SubRegex {
            Concatenation {
                Piece1 {}
                Piece2 {}
                PieceN {}
            }
        }
        [No Quantifier!!!]
    }
    [...]

    =>

    [...]
    Piece1 {}
    Piece2 {}
    PieceN {}
    [...]

 */
DEFINE_REWRITE_PATTERN_MACRO(SimplifySubregexNotQuantified,
                             RegexParser::dialect::SubRegexOp);

/*
 * Simplify a subregex which contains a single concatenation with single piece,
 however only one between the subregex and the piece can be quantified.

    [...]
    Piece {
        SubRegex {
            Concatenation {
                Piece1 {
                    Atom
                    [Quantifier1]
                }
            }
        }
        [Quantifier2]
    }
    [...]

    =>

    [...]
    Piece {
        Atom
        [Quantifier1 | Quantifier2] <- if they are both present cannot optimize
    }
    [...]
 */
DEFINE_REWRITE_PATTERN_MACRO(SimplifySubregexSinglePiece,
                             RegexParser::dialect::SubRegexOp);

/*
 * When RootOp `hasSuffix` is true, or its children ConcatenationOp `hasSuffix`
 * is true, if the last piece of such ConcatenationOp has a quantifier, then
 * assign quantifier.min = quantifier.max. If quantifier.min == 0, then this
 * piece can be removed.
 *
 * For example:
 * `ab{10,20}|cd{30,40}|ef{5,7}$` -> `ab{10,10}|cd{30,30}|ef{5,7}`
 * `abcd*|efgh+` -> `abc|efgh`
 */
DEFINE_REWRITE_PATTERN_MACRO(SimplifyLeadingQuantifiers,
                             RegexParser::dialect::RootOp);

} // namespace RegexParser::passes