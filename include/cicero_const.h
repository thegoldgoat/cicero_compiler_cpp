#pragma once

enum CiceroOpCodes {
    ACCEPT = 0,
    SPLIT = 1,
    MATCH_CHAR = 2,
    JUMP = 3,
    END_WITHOUT_ACCEPTING = 4,
    MATCH_ANY = 5,
    ACCEPT_PARTIAL = 6,
    NOT_MATCH_CHAR = 7,
};

enum CiceroBinaryOutputFormat { Binary, Hex };

enum CiceroAction {
    None,
    DumpRegexMLIR,
    DumpCiceroMLIR,
    DumpCiceroDOT,
    DumpCompiled
};

#include <string>

const std::string CICERO_COLOR_MATCH = "#ffa822";
const std::string CICERO_COLOR_SPLIT = "#dee0e6";
const std::string CICERO_COLOR_ACCEPT = "#1ac0c6";
const std::string CICERO_COLOR_JUMP = "#2792ce";