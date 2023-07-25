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
 * with same prefix, for example `this|that` -> `th(is|at)` but also
 * `this|that|th` -> `th(is|at)?`.
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
 * We can simplify the leading piece of each concatenation that contains a
 * quantifier, and replace that quantifier max with newmax=min
 *
 * For example:
 * `ab{10,20}|cd{30,40}` -> `ab{10,10}|cd{30,30}`
 * `abcd*|efgh+` -> `abc|efgh`
 */
DEFINE_REWRITE_PATTERN_MACRO(SimplifyLeadingQuantifiers,
                             RegexParser::dialect::RootOp);

} // namespace RegexParser::passes