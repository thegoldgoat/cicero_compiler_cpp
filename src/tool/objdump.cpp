#include "cicero_const.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

const int BITS_INSTR_TYPE = 3;
const int BITS_INSTR_DATA = 13;
const int BITS_INSTR = BITS_INSTR_TYPE + BITS_INSTR_DATA;

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("Usage: %s <input file>\n", argv[0]);
        return -1;
    }

    FILE *fp = fopen(argv[1], "rb");

    if (!fp) {
        printf("Error opening file %s\n", argv[1]);
        return -1;
    }

    int pc = 0;
    uint16_t instruction;
    while (fread(&instruction, sizeof(instruction), 1, fp)) {
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

        pc += 1;
    }
}