#pragma once

#include <memory>
#include <vector>

#include <iostream>
#include <optional>
#include <string>

namespace RegexParser::AST {

using namespace std;

// Forward declarations
class RegExp;
class Quantifier;

// Base Abstract class for an AST node
class Node {
  public:
};

// AST node for an atom, which is a character or a subexpression
class Atom : Node {
  public:
};

class AnyChar : public Atom {};

class Char : public Atom {
  public:
    Char(char c) : c(c) {}

  private:
    char c;
};

class SubExpression : public Atom {
  public:
    SubExpression(unique_ptr<RegExp> regExp) : regExp(move(regExp)) {}

  private:
    unique_ptr<RegExp> regExp;
};

class Group : public Atom {
  public:
    Group(vector<bool> &&charsToMatch) : charsToMatch(move(charsToMatch)) {}

  private:
    vector<bool> charsToMatch;
};

enum QuantifierType { STAR, PLUS, OPTIONAL, RANGE };

// Abstract AST node for a quantifier
class Quantifier : Node {
  public:
    static unique_ptr<Quantifier> buildStarQuantifier() {
        return make_unique<Quantifier>(QuantifierType::STAR, 0, -1);
    }

    static unique_ptr<Quantifier> buildPlusQuantifier() {
        return make_unique<Quantifier>(QuantifierType::PLUS, 1, -1);
    }

    static unique_ptr<Quantifier> buildOptionalQuantifier() {
        return make_unique<Quantifier>(QuantifierType::OPTIONAL, 0, 1);
    }

    static unique_ptr<Quantifier> buildRangeQuantifier(int min, int max) {
        return make_unique<Quantifier>(QuantifierType::RANGE, min, max);
    }

    Quantifier(QuantifierType type, int min, int max)
        : type(type), min(min), max(max) {}

  private:
    QuantifierType type;
    int min;
    int max;
};

// AST node for a "piece", which is an atom with an optional quantifier
class Piece : Node {
  public:
    Piece(unique_ptr<Atom> atom, unique_ptr<Quantifier> quantifier)
        : atom(move(atom)), quantifier(move(quantifier)) {}

    bool hasQuantifier() { return quantifier != nullptr; }

  private:
    unique_ptr<Atom> atom;
    unique_ptr<Quantifier> quantifier;
};

// AST node for a concatenation of "pieces"
class Concatenation : Node {
  public:
    Concatenation(vector<unique_ptr<Piece>> pieces) : pieces(move(pieces)) {}

  private:
    vector<unique_ptr<Piece>> pieces;
};

// AST node for a regular expression (or subexpression)
class RegExp : Node {
  public:
    RegExp(vector<unique_ptr<Concatenation>> &&concatenations)
        : concatenations(move(concatenations)) {}

  private:
    vector<unique_ptr<Concatenation>> concatenations;
};

} // namespace RegexParser::AST