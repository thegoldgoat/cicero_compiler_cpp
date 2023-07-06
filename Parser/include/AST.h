#pragma once

#include <memory>
#include <vector>

#include <iostream>
#include <optional>
#include <string>

namespace RegexParser::AST {

using namespace std;

// Base Abstract class for an AST node
class Node {
  public:
};

// AST node for an atom, which is a character or a subexpression
class Atom : Node {
  public:
};

enum QuantifierType { STAR, PLUS, OPTIONAL, RANGE };

// Abstract AST node for a quantifier
class Quantifier : Node {
  public:
    static Quantifier buildStarQuantifier() {
        return Quantifier(QuantifierType::STAR, 0, -1);
    }

    static Quantifier buildPlusQuantifier() {
        return Quantifier(QuantifierType::PLUS, 1, -1);
    }

    static Quantifier buildOptionalQuantifier() {
        return Quantifier(QuantifierType::OPTIONAL, 0, 1);
    }

    static Quantifier buildRangeQuantifier(int min, int max) {
        return Quantifier(QuantifierType::RANGE, min, max);
    }

    ~Quantifier() { cout << "Quantifier destructor called" << endl; }

  private:
    Quantifier(QuantifierType type, int min, int max)
        : type(type), min(min), max(max) {
        cout << "Quantifier constructor called" << endl;
    }

    QuantifierType type;
    int min;
    int max;
};

// AST node for a "piece", which is an atom with an optional quantifier
class Piece : Node {
  public:
    Piece(Atom atom, optional<Quantifier> quantifier)
        : atom(atom), quantifier(quantifier) {}

    bool hasQuantifier() { return quantifier.has_value(); }

  private:
    Atom atom;
    optional<Quantifier> quantifier;
};

// AST node for a concatenation of "pieces"
class Concatenation : Node {
  public:
    Concatenation(vector<Piece> pieces) : pieces(move(pieces)) {}

  private:
    vector<Piece> pieces;
};

// AST node for a regular expression (or subexpression)
class RegExp : Node {
  public:
    RegExp(vector<Concatenation> &&concatenations)
        : concatenations(move(concatenations)) {}

  private:
    vector<Concatenation> concatenations;
};

} // namespace RegexParser::AST