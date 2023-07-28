#pragma once

#include "CiceroDialectWrapper.h"
#include "cicero_macros.h"

#include <vector>

namespace cicero_compiler::passes {

/// @brief Helper class that collects the followers of a split operation
/// @details This class is used to collect the followers of a split operation
/// in a way that easily enables merge optimizations of said followers, by
/// grouping together same-type operations e.g. MatchCharOp on same chars, and
/// MatchAny
class SplitFollowers {
  public:
    SplitFollowers() {}

    void addFollowers(dialect::FlatSplitOp &op) {
        _getSplitFollowers(op.getOperation());
    }

    void dump() {
        llvm::outs() << "---- MatchCharOp Followers ----\n";
        for (auto &matchChar : matchChars) {
            for (auto &matchCharOp : matchChar) {
                matchCharOp.dump();
            }
        }

        llvm::outs() << "---- MatchAny Followers ----\n";
        for (auto &matchAnyOp : matchAnys) {
            matchAnyOp.dump();
        }
    }

  private:
    void _getSplitFollowers(mlir::Operation *op) {
        if (!op) {
            return;
        }

        if (CAST_MACRO(splitOp, op, dialect::FlatSplitOp)) {
            _getSplitFollowers(op->getNextNode());

            auto splitTargetOp = mlir::SymbolTable::lookupNearestSymbolFrom(
                op, splitOp.getSplitTargetAttr());
            _getSplitFollowers(splitTargetOp);
        } else if (CAST_MACRO(matchCharOp, op, dialect::MatchCharOp)) {
            matchChars[matchCharOp.getTargetChar()].emplace_back(
                std::move(matchCharOp));
        } else if (CAST_MACRO(matchAnyOp, op, dialect::MatchAnyOp)) {
            matchAnys.emplace_back(std::move(matchAnyOp));
        }
    }

    std::vector<dialect::MatchCharOp> matchChars[256];
    std::vector<dialect::MatchAnyOp> matchAnys;
};

} // namespace cicero_compiler::passes