#include <iostream>
#include "CSVParser.h"

using namespace std;

const string tmpDir = "/tmp/";

int main() {
    auto input = parse_csv(REGEX_CSV_PATH);
    for (auto row : input) {
        for (auto cell : row) {
            cout << cell << ",";
        }
        cout << endl;
    }
    return 0;
}