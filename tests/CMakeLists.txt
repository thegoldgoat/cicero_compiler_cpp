set(REGEX_CSV_PATH ${PROJECT_BINARY_DIR}/tests/regex_tests.csv)

add_custom_command(
OUTPUT
    ${REGEX_CSV_PATH}
DEPENDS
    data/generate_csv.py
COMMAND
    python3
    ${CMAKE_CURRENT_SOURCE_DIR}/data/generate_csv.py
    ${REGEX_CSV_PATH}
)

add_executable(
    compile_and_run
    src/compile_and_run.cpp
    src/CSVParser.h
    ${REGEX_CSV_PATH}
)

add_test(
NAME
    compile_and_run
COMMAND
    compile_and_run
)

target_compile_definitions(
    compile_and_run
    PRIVATE
    REGEX_CSV_PATH="${REGEX_CSV_PATH}"
)

add_executable(
    CSVParser_test
    src/test_csv_parser.cpp
)

add_test(
NAME
    CSVParser_test
COMMAND
    CSVParser_test
)

target_compile_definitions(
    CSVParser_test
    PRIVATE
    CSVs_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/CSVParser"
)