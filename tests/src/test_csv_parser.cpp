#include "CSVParser.h"
#include <cassert>
#include <iostream>

void test_parse_csv(std::vector<std::vector<std::string>> &expected,
                    std::string inputPath) {
    std::vector<std::vector<std::string>> actual = parse_csv(inputPath);
    assert(expected == actual);
}

void test_parse_csv_invalid(std::string inputPath) {
    try {
        parse_csv(inputPath);
        assert(false && "Expected exception not thrown");
    } catch (const std::runtime_error &e) {
        // Expected exception thrown
    }
}

int main() {

    std::string base_csv_path = std::string(CSVs_PATH);

    std::vector<std::vector<std::string>> expected1 = {
        {"Name", "Age", "City"},
        {"Alice", "25", "New York"},
        {"Bob", "30", "San Francisco"},
        {"Charlie", "35", "Seattle"}};
    std::vector<std::vector<std::string>> expected2 = {
        {"[a-z][a-zA-Z0-9_]*[a-z]{2,5}h+", "byujhh6", "1"},
        {"[a-z][a-zA-Z0-9_]*[a-z]{2,5}h+", "byujhh6", "1", ""},
        {"", "", ""}};
    test_parse_csv(expected1, base_csv_path + "/1.csv");
    std::cout << "Test1 OK\n";
    test_parse_csv(expected2, base_csv_path + "/2.csv");
    std::cout << "Test2 OK\n";
    test_parse_csv_invalid(base_csv_path + "/3.csv");
    std::cout << "Test3 OK\n";
    test_parse_csv_invalid(base_csv_path + "/4.csv");
    std::cout << "Test4 OK\n";
    std::cout << "All tests passed\n";
    return 0;
}