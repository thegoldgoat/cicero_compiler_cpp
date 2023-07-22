set(REGEX_CSV_PATH ${PROJECT_BINARY_DIR}/tests/regex_tests.csv)
set(REGEX_COMPILED_DIR_PATH ${PROJECT_BINARY_DIR}/tests/regex_compiled)

add_custom_command(
OUTPUT
    ${REGEX_CSV_PATH}
DEPENDS
    data/generate_csv.py
    data/regexes.txt
    data/inputs.txt
    ciceroc
COMMAND
    python3
    ${CMAKE_CURRENT_SOURCE_DIR}/data/generate_csv.py
    ${REGEX_CSV_PATH}
    ${REGEX_COMPILED_DIR_PATH}
    $<TARGET_FILE:ciceroc>
)

add_executable(
    run_on_simulator
    src/run_on_simulator.cpp
    src/CSVParser.h
    ${REGEX_CSV_PATH}
)

target_include_directories(
    run_on_simulator
    PRIVATE
    ../thirdparty/SoftwareCicero/include
)

target_link_libraries(
    run_on_simulator
    PRIVATE
    CiceroMulti)

add_test(
NAME
    run_on_simulator
COMMAND
    run_on_simulator
)

target_compile_definitions(
    run_on_simulator
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