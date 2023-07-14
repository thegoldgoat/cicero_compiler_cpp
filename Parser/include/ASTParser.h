#pragma once

#include <AST.h>
#include <fstream>
#include <iostream>
#include <memory>

namespace RegexParser {

std::unique_ptr<AST::RegExp> parseRegexFromFile(const std::string &regexPath);
std::unique_ptr<AST::RegExp> parseRegexFromString(const std::string &regexPath);

} // namespace RegexParser
