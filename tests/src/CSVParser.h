#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::vector<std::string>> parse_csv(const std::string &filename) {
    std::vector<std::vector<std::string>> rows;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream cell;
        // Remove trailing '\n' or '\r'
        if (line[line.size() - 1] == '\n' || line[line.size() - 1] == '\r') {
            line.pop_back();
        }

        bool withinQuotes = false;
        for (std::size_t i = 0; i < line.size() - 1; i++) {
            switch (line[i]) {
            case '"':
                if (withinQuotes) {
                    // If the next char is a quote, then this quote is escaped
                    // Skip next char (i++) and add a quote to the cell
                    if (line[i + 1] == '"') {
                        cell << '"';
                        i++;
                    } else {
                        if (line[i+1] != ',' || i == line.size() - 2) {
                            throw std::runtime_error(
                                "Invalid CSV file, endquote not followed by "
                                "comma nor at end of the line");
                        }
                        i++;
                        withinQuotes = false;
                        row.push_back(cell.str());
                        cell.str("");
                    }
                } else {
                    if (cell.str().empty()) {
                        withinQuotes = true;
                    } else {
                        throw std::runtime_error(
                            "Invalid CSV file, quote in the middle of a cell "
                            "not quoted at beginning");
                    }
                }
                break;
            case ',':
                if (withinQuotes) {
                    cell << ',';
                } else {
                    row.push_back(cell.str());
                    cell.str("");
                }
                break;
            default:
                cell << line[i];
                break;
            }
        }

        switch (line.back()) {
        case '"':
            if (withinQuotes) {
                withinQuotes = false;
                row.push_back(cell.str());
                cell.str("");
            } else {
                throw std::runtime_error("Invalid CSV file, quote at end of "
                                         "line not quoted at beginning");
            }
            break;
        case ',':
            if (withinQuotes) {
                throw std::runtime_error("Invalid CSV file, comma at end of "
                                         "line, quoted at beginning");
            } else {
                row.push_back(cell.str());
                cell.str("");
                row.push_back("");
            }
            break;
        default:
            if (withinQuotes) {
                throw std::runtime_error("Invalid CSV file, last cell not "
                                         "closed with quote");
            } else {
                cell << line.back();
                row.push_back(cell.str());
                cell.str("");
            }
            break;
        }

        rows.push_back(row);
    }
    return rows;
}