#pragma once

#include <AST.h>
#include <fstream>
#include <iostream>
#include <memory>

namespace RegexParser {

std::unique_ptr<AST::Root> parseRegexFromFile(const std::string &regexPath);
std::unique_ptr<AST::Root> parseRegexFromString(const std::string &regexPath);

} // namespace RegexParser
