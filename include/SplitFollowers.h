#pragma once

#include "CiceroDialectWrapper.h"
#include "cicero_helper.h"
#include "cicero_macros.h"

#include <unordered_set>
#include <vector>

namespace cicero_compiler::passes {

unsigned int splitFollowersSymbolCounter = 0;

/// @brief Helper class that collects the followers of a split operation
/// @details This class is used to collect the followers of a split operation
/// in a way that easily enables merge optimizations of said followers, by
/// grouping together same-type operations e.g. MatchCharOp on same chars, and
/// MatchAny
class SplitFollowers {
  public:
    /// @brief Factorize followers of a split operation
    /// @param rewriter
    void optimizeByFactorize(dialect::FlatSplitOp &op,
                             mlir::PatternRewriter &rewriter) {
        auto moduleOp =
            mlir::dyn_cast<mlir::ModuleOp>(op.getOperation()->getParentOp());
        populateSplitFollowers(op.getOperation());

        for (auto &matchChar : matchChars) {
            if (matchChar.size() < 2) {
                continue;
            }
            llvm::outs() << "Begin optimization for character = "
                         << matchChar[0].getTargetChar() << "\n";
            cicero_compiler::dumpCiceroDot(moduleOp);
            applyOptimization<dialect::MatchCharOp>(op, rewriter, matchChar);
            llvm::outs() << "End of optimization: \n";
            cicero_compiler::dumpCiceroDot(moduleOp);
            return;
        }

        if (matchAnys.size() < 2) {
            return;
        }
        llvm::outs() << "Optimizing followers MatchAnyOps\n";
        // applyOptimization<dialect::MatchAnyOp>(op, rewriter, matchAnys);
    }

  private:
    template <typename OpsT>
    void applyOptimization(dialect::FlatSplitOp ancestorSplitOp,
                           mlir::PatternRewriter &rewriter,
                           std::vector<OpsT> &ops) {
        assert(ops.size() > 1 &&
               "Must not try to apply optimization on less than 2 ops");

        auto moduleOp = mlir::dyn_cast<mlir::ModuleOp>(
            ancestorSplitOp.getOperation()->getParentOp());

        ancestorSplitOp.getOperation()->getParentOp()->dump();
        llvm::outs() << "applyOptimizations on " << ancestorSplitOp;

        // Step 0: clone the operation after ancestorSplitOp
        rewriter.setInsertionPoint(ancestorSplitOp);

        if (ancestorSplitOp->hasAttr(mlir::SymbolTable::getSymbolAttrName())) {
            auto oldAncestorSymbol = ancestorSplitOp.getName();
            auto newAncestorSymbol = getNewSymbolName();
            mlir::SymbolTable::setSymbolName(ancestorSplitOp,
                                             newAncestorSymbol);
            auto splitBeforeAncestor = rewriter.create<dialect::FlatSplitOp>(
                ancestorSplitOp->getLoc(), newAncestorSymbol);
            splitBeforeAncestor.setName(oldAncestorSymbol);
        } else {
            auto newAncestorSymbol = getNewSymbolName();
            ancestorSplitOp.setName(newAncestorSymbol);
            rewriter.create<dialect::FlatSplitOp>(ancestorSplitOp->getLoc(),
                                                  newAncestorSymbol);
        }
        mlir::Operation *opAfterAncestor =
            rewriter.clone(*ops[0].getOperation());
        opAfterAncestor->removeAttr(mlir::SymbolTable::getSymbolAttrName());

        llvm::outs() << "Step 0 finished\n";
        cicero_compiler::dumpCiceroDot(moduleOp);

        // Step 1: Remove all the ops except the first one (preserve control
        // flow). Also, "backup" the symbols of the followers of the deleted ops
        std::vector<std::string> symbolsOfFollowersOfDeletedOps;
        symbolsOfFollowersOfDeletedOps.reserve(ops.size() - 1);

        for (auto op : ops) {
            mlir::Operation *opToRemove = op.getOperation();
            mlir::Operation *followerOfOpToRemove = opToRemove->getNextNode();

            if (!followerOfOpToRemove) {
                ancestorSplitOp.emitError("applyOptimization: follower of op "
                                          "to remove is null??? How can it be "
                                          "possible for MatchChar|MatchAny ??");
                throw std::runtime_error("");
            }

            if (followerOfOpToRemove->hasAttr(
                    mlir::SymbolTable::getSymbolAttrName())) {
                symbolsOfFollowersOfDeletedOps.emplace_back(
                    mlir::SymbolTable::getSymbolName(followerOfOpToRemove)
                        .getValue()
                        .str());
            } else if (opToRemove->hasAttr(
                           mlir::SymbolTable::getSymbolAttrName())) {
                symbolsOfFollowersOfDeletedOps.emplace_back(
                    mlir::SymbolTable::getSymbolName(opToRemove)
                        .getValue()
                        .str());
            } else {
                auto newSymbol = getNewSymbolName();
                mlir::SymbolTable::setSymbolName(followerOfOpToRemove,
                                                 newSymbol);
                symbolsOfFollowersOfDeletedOps.emplace_back(newSymbol);
            }

            // "Disconnect" the flatsplit that points to opToRemove:
            // 1. The parent flatsplit is the prevOp
            // 2. The parent is the user of my symbol

            bool parentIsPrevOp;
            auto parentSplit = findSplitParent(opToRemove, &parentIsPrevOp);

            if (parentIsPrevOp) {
                if (parentSplit->hasAttr(
                        mlir::SymbolTable::getSymbolAttrName())) {
                    auto prevSymbol = parentSplit.getName().str();
                    auto splitTarget = parentSplit.getSplitTarget().str();
                    // Why it crashes on this?!?!?!
                    rewriter.setInsertionPointAfter(parentSplit);
                    auto replacedPrev =
                        rewriter.replaceOpWithNewOp<dialect::JumpOp>(
                            parentSplit, splitTarget);
                    replacedPrev.setName(prevSymbol);
                } else {
                    rewriter.setInsertionPointAfter(parentSplit);
                    rewriter.replaceOpWithNewOp<dialect::JumpOp>(
                        parentSplit, parentSplit.getSplitTarget());
                }
            } else {
                if (removeOperationAndMoveSymbolToNext(parentSplit, rewriter)
                        .failed()) {
                    ancestorSplitOp.emitError(
                        "applyOptimization: failed to remove "
                        "one of the flatsplit that are users of "
                        "opToDelete???");
                    throw std::runtime_error("");
                }
            }

//            auto prevOpToRemove = opToRemove->getPrevNode();
//            auto flatPrev =
//                mlir::dyn_cast<dialect::FlatSplitOp>(prevOpToRemove);
//            if (flatPrev) {
//                if (flatPrev->hasAttr(mlir::SymbolTable::getSymbolAttrName())) {
//                    auto prevSymbol = flatPrev.getName().str();
//                    auto splitTarget = flatPrev.getSplitTarget().str();
//                    // Why it crashes on this?!?!?!
//                    rewriter.setInsertionPointAfter(flatPrev);
//                    auto replacedPrev =
//                        rewriter.replaceOpWithNewOp<dialect::JumpOp>(
//                            flatPrev, splitTarget);
//                    replacedPrev.setName(prevSymbol);
//                } else {
//                    rewriter.setInsertionPointAfter(flatPrev);
//                    rewriter.replaceOpWithNewOp<dialect::JumpOp>(
//                        flatPrev, flatPrev.getSplitTarget());
//                }
//            } else {
//                auto user = findSplitParent(opToRemove);
//
//                if (removeOperationAndMoveSymbolToNext(user, rewriter)
//                        .failed()) {
//                    ancestorSplitOp.emitError(
//                        "applyOptimization: failed to remove "
//                        "one of the flatsplit that are users of "
//                        "opToDelete???");
//                    throw std::runtime_error("");
//                }
//            }

            if (removeOperationAndMoveSymbolToNext(opToRemove, rewriter)
                    .failed()) {
                ancestorSplitOp.emitError("applyOptimization: failed to remove "
                                          "one of the duplicated operations");
                throw std::runtime_error("");
            }
        }

        llvm::outs() << "Step 1 finished\n";
        cicero_compiler::dumpCiceroDot(moduleOp);

        // Step 2: Populate the flow of execution after the factorized op

        rewriter.setInsertionPointAfter(opAfterAncestor);

        for (size_t i = 0; i < symbolsOfFollowersOfDeletedOps.size() - 1; i++) {
            rewriter.create<dialect::FlatSplitOp>(
                rewriter.getUnknownLoc(), symbolsOfFollowersOfDeletedOps[i]);
        }
        rewriter.create<dialect::JumpOp>(rewriter.getUnknownLoc(),
                                         symbolsOfFollowersOfDeletedOps.back());
    }

    static std::string getNewSymbolName() {
        return "SFO" + std::to_string(splitFollowersSymbolCounter++);
    }

    dialect::FlatSplitOp
    findSplitParent(mlir::Operation *op,
                    bool *splitFoundIsImmediatePredecessor) {

        auto opBefore = op->getPrevNode();
        assert(opBefore &&
               "op should never beopBefore the absolute first one??");
        if (CAST_MACRO(splitOp, opBefore, dialect::FlatSplitOp)) {
            *splitFoundIsImmediatePredecessor = true;
            return splitOp;
        }

        auto usesOptional =
            mlir::SymbolTable::getSymbolUses(op, op->getParentOp());
        assert(usesOptional.has_value() &&
               "findSplitParent: cannot find usages of op");
        auto uses = usesOptional.value();
        auto usesVector =
            std::vector<mlir::SymbolTable::SymbolUse>(uses.begin(), uses.end());
        if (usesVector.size() != 1) {
            op->emitError("findSplitParent: Should have found only "
                          "one use for symbol of op, found ")
                << usesVector.size() << " instead: " << op;
            throw std::runtime_error("");
        }
        auto user = usesVector[0].getUser();

        if (CAST_MACRO(jumpOp, user, dialect::JumpOp)) {
            // User is another jump, recursive call
            return findSplitParent(user, splitFoundIsImmediatePredecessor);
        } else if (CAST_MACRO(splitOp, user, dialect::FlatSplitOp)) {
            // Found the split by following symbol!
            *splitFoundIsImmediatePredecessor = false;
            return splitOp;
        }

        op->emitError(
            "findSplitParent: symbol user was not a jump not split???")
            << user;
        throw std::runtime_error("");
    }

    std::unordered_set<mlir::Operation *> operationsVisited;

    void populateSplitFollowers(mlir::Operation *op) {
        if (!op) {
            return;
        }

        // Insert into the visited set. If op is already in the set, then return
        if (operationsVisited.insert(op).second == false) {
            return;
        }

        if (CAST_MACRO(splitOp, op, dialect::FlatSplitOp)) {
            populateSplitFollowers(op->getNextNode());

            auto splitTargetOp = mlir::SymbolTable::lookupNearestSymbolFrom(
                op, splitOp.getSplitTargetAttr());
            populateSplitFollowers(splitTargetOp);
        } else if (CAST_MACRO(matchCharOp, op, dialect::MatchCharOp)) {
            matchChars[matchCharOp.getTargetChar()].emplace_back(matchCharOp);
        } else if (CAST_MACRO(matchAnyOp, op, dialect::MatchAnyOp)) {
            matchAnys.emplace_back(matchAnyOp);
        } else if (CAST_MACRO(jumpOp, op, dialect::JumpOp)) {
            mlir::Operation *jumpTargetOp =
                mlir::SymbolTable::lookupNearestSymbolFrom(
                    op->getParentOp(), jumpOp.getTargetAttr());
            populateSplitFollowers(jumpTargetOp);
        }
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

    std::vector<dialect::MatchCharOp> matchChars[256];
    std::vector<dialect::MatchAnyOp> matchAnys;
};

} // namespace cicero_compiler::passes