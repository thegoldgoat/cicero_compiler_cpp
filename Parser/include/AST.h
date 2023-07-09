#pragma once

#include <memory>
#include <vector>

#include <iostream>
#include <optional>
#include <string>

#define ADDR_TO_STR(addr) to_string(reinterpret_cast<uintptr_t>(addr))

namespace RegexParser::AST {

using namespace std;

// Forward declarations
class RegExp;
class Quantifier;

// Base Abstract class for an AST node
class Node {
  public:
    Node(const string &&nodeName) : nodeName(nodeName) {}

    virtual string toDotty() const {
        return ADDR_TO_STR(this) + " [label=\"" + nodeName + "\"];\n";
    }

  protected:
    const string nodeName;
};

// AST node for an atom, which is an abstract class for terminal nodes or
// subregex
class Atom : public Node {
  public:
    Atom(const string &&nodeName) : Node(std::move(nodeName)) {}
};

class AnyChar : public Atom {
  public:
    AnyChar() : Atom("AnyChar") {}
};

class Char : public Atom {
  public:
    Char(char c) : Atom(string("Char: ") + c), c(c) {}

  private:
    char c;
};

class SubRegex : public Atom {
  public:
    SubRegex(unique_ptr<RegExp> regExp)
        : Atom("SubRegex"), regExp(move(regExp)) {}

    string toDotty() const override;

  private:
    unique_ptr<RegExp> regExp;
};

class Group : public Atom {
  public:
    Group(vector<bool> &&charsToMatch)
        : Atom("Group"), charsToMatch(move(charsToMatch)) {}

  private:
    vector<bool> charsToMatch;
};

enum QuantifierType { STAR, PLUS, OPTIONAL, RANGE };

// Abstract AST node for a quantifier
class Quantifier : public Node {
  public:
    static unique_ptr<Quantifier> buildStarQuantifier() {
        return make_unique<Quantifier>(
            Quantifier(QuantifierType::STAR, 0, -1, "*"));
    }

    static unique_ptr<Quantifier> buildPlusQuantifier() {
        return make_unique<Quantifier>(
            Quantifier(QuantifierType::PLUS, 1, -1, "+"));
    }

    static unique_ptr<Quantifier> buildOptionalQuantifier() {
        return make_unique<Quantifier>(
            Quantifier(QuantifierType::OPTIONAL, 0, 1, "?"));
    }

    static unique_ptr<Quantifier> buildRangeQuantifier(int min, int max) {
        return make_unique<Quantifier>(
            Quantifier(QuantifierType::RANGE, min, max,
                       "{" + to_string(min) + "," + to_string(max) + "}"));
    }

  private:
    Quantifier(QuantifierType type, int min, int max, string &&nodeNameTag)
        : Node("Quantifier: " + nodeNameTag), type(type), min(min), max(max) {}
    QuantifierType type;
    int min;
    int max;
};

// AST node for a "piece", which is an atom with an optional quantifier
class Piece : public Node {
  public:
    Piece(unique_ptr<Atom> atom, unique_ptr<Quantifier> quantifier)
        : Node("Piece"), atom(move(atom)), quantifier(move(quantifier)) {}

    bool hasQuantifier() { return quantifier != nullptr; }

    string toDotty() const override {
        string dotty = Node::toDotty();
        dotty += ADDR_TO_STR(this) + " -> " + ADDR_TO_STR(atom.get()) + ";\n" +
                 atom->toDotty();
        if (quantifier != nullptr) {
            dotty += ADDR_TO_STR(this) + " -> " +
                     ADDR_TO_STR(quantifier.get()) + ";\n" +
                     quantifier->toDotty();
        }
        return dotty;
    }

  private:
    unique_ptr<Atom> atom;
    unique_ptr<Quantifier> quantifier;
};

// AST node for a concatenation of "pieces"
class Concatenation : public Node {
  public:
    Concatenation(vector<unique_ptr<Piece>> pieces)
        : Node("Concatenation"), pieces(move(pieces)) {}

    string toDotty() const override {
        string dotty = Node::toDotty();
        for (auto &piece : pieces) {
            dotty += ADDR_TO_STR(this) + " -> " + ADDR_TO_STR(piece.get()) +
                     ";\n" + piece->toDotty();
        }
        return dotty;
    }

  private:
    vector<unique_ptr<Piece>> pieces;
};

// AST node for a regular expression (or SubRegex)
class RegExp : public Node {
  public:
    RegExp(vector<unique_ptr<Concatenation>> &&concatenations)
        : Node("RegExp"), concatenations(move(concatenations)) {}

    string toDotty() const override {
        string dotty = Node::toDotty();
        for (auto &concatenation : concatenations) {
            dotty += ADDR_TO_STR(this) + " -> " +
                     ADDR_TO_STR(concatenation.get()) + ";\n" +
                     concatenation->toDotty();
        }
        return dotty;
    }

  private:
    vector<unique_ptr<Concatenation>> concatenations;
};

inline string SubRegex::toDotty() const {
    return Node::toDotty() + ADDR_TO_STR(this) + " -> " +
           ADDR_TO_STR(regExp.get()) + ";\n" + regExp->toDotty();
}

} // namespace RegexParser::AST