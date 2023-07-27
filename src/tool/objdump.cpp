#include "cicero_const.h"
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <stdio.h>

const int BITS_INSTR_TYPE = 3;
const int BITS_INSTR_DATA = 13;
const int BITS_INSTR = BITS_INSTR_TYPE + BITS_INSTR_DATA;

using namespace std;

void dumpInstruction(uint16_t instruction, uint16_t &pc);

int main(int argc, char **argv) {
    if (argc == 1) {
        cerr << "Usage: " << argv[0] << " <input file> [binary | hex]" << endl;
        cerr << "Specify binary or hex to indicate the input format, binary is "
                "default"
             << endl;
        return -1;
    }

    string inputFilename = argv[1];
    if (inputFilename == "-") {
        inputFilename = "/dev/stdin";
    }
    CiceroBinaryOutputFormat binaryInputFormat;

    if (argc < 3) {
        binaryInputFormat = Binary;
    } else {
        string inputFormat = argv[2];
        if (inputFormat == "binary") {
            binaryInputFormat = Binary;
        } else if (inputFormat == "hex") {
            binaryInputFormat = Hex;
        } else {
            cerr << "Unknown input format: " << inputFormat
                 << "; it must be either 'binary' or 'hex'" << endl;
            return -1;
        }
    }

    if (binaryInputFormat == Binary) {
        auto fp = ifstream(inputFilename, ios::binary);
        if (!fp) {
            cerr << "Error opening file " << inputFilename << endl;
            return -1;
        }

        uint16_t instruction, pc = 0;
        while (fp.read(reinterpret_cast<char *>(&instruction),
                       sizeof(instruction))) {
            dumpInstruction(instruction, pc);
        }
    } else {
        auto fp = ifstream(inputFilename);
        if (!fp) {
            cerr << "Error opening file " << inputFilename << endl;
            return -1;
        }

        uint16_t instruction, pc = 0;
        string line;
        while (getline(fp, line)) {
            instruction = stoi(line, nullptr, 16);

            dumpInstruction(instruction, pc);
        }
    }
}

void dumpInstruction(uint16_t instruction, uint16_t &pc) {
    printf("%03d: %04x - ", pc, instruction);
    int type = instruction >> BITS_INSTR_DATA;
    int data = instruction & 0x1fff;
    switch (type) {
    case CiceroOpCodes::ACCEPT:
        printf("ACCEPT\n");
        break;
    case CiceroOpCodes::SPLIT:
        printf("SPLIT\t {%d,%d} \n", pc + 1, data);
        break;
    case CiceroOpCodes::MATCH_CHAR:
        printf("MATCH\t char %c\n", data);
        break;
    case CiceroOpCodes::JUMP:
        printf("JMP to \t %d \n", data);
        break;
    case CiceroOpCodes::END_WITHOUT_ACCEPTING:
        printf("END_WITHOUT_ACCEPTING\n");
        break;
    case CiceroOpCodes::MATCH_ANY:
        printf("MATCH_ANY\n");
        break;
    case CiceroOpCodes::ACCEPT_PARTIAL:
        printf("ACCEPT_PARTIAL\n");
        break;
    case CiceroOpCodes::NOT_MATCH_CHAR:
        printf("NOT_MATCH\t char %c\n", data);
        break;
    default:
        printf("UNKNOWN %d\t data %d\n", type, data);
        break;
    }

    pc++;
}