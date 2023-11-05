#include "CSVParser.h"
#include "CiceroMulti.h"
#include <cassert>
#include <iostream>

using namespace std;

void run(bool withOptimizations) {
    auto input = parse_csv(REGEX_CSV_PATH);
    Cicero::CiceroMulti cicero;

    string lastProgram;
    auto it = input.begin();
    ++it; // skip first line of CSV that contains headers
    size_t completed = 0;
    for (; it != input.end(); ++it) {
        cerr << "\r" << completed++ << "/" << input.size() - 1 << flush;
        auto row = *it;
        // Assert row contains 3 elements
        if (row.size() != 3) {
            cerr << "Test failed, row from CSV does not have 3 element, has "
                 << row.size() << endl;
            for (auto element : row) {
                cerr << element << endl;
            }
            assert(false);
        }

        string program = row[0];
        string input = row[1];
        string expectedAsString = row[2];

        if (expectedAsString != "1" && expectedAsString != "0") {
            cerr << "Test failed, expected value is not 1 or 0, but is "
                 << expectedAsString << endl;
            assert(false);
        }

        bool expected = expectedAsString == "1";

        if (program != lastProgram) {
            if (withOptimizations) {
                program += "_optimized";
            }
            cicero.setProgram(program.c_str());
            lastProgram = program;
        }

        bool actual = cicero.match(input);

        if (actual != expected) {
            cerr << "Test failed: " << program << " " << input << " "
                 << expectedAsString
                 << "; withOptimizations = " << withOptimizations << endl;
            assert(false);
        }
    }
}

int main() {
    run(false);
    run(true);
    return 0;
}