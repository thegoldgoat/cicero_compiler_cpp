import csv
import os
import re
import sys
import subprocess

if len(sys.argv) != 4:
    print('Usage: python3 generate_csv.py <output_file> <compiled_regexes_path> <path_to_compiler>')
    sys.exit(1)

results_path = sys.argv[1]
compiled_regexes_path = sys.argv[2]
compiler_path = sys.argv[3]

# create the compiled regexes directory if it doesn't exist
if not os.path.exists(compiled_regexes_path):
    os.makedirs(compiled_regexes_path)

# Get the absolute path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

regexes_path = os.path.join(script_dir, 'regexes.txt')
inputs_path = os.path.join(script_dir, 'inputs.txt')

# Open file1 and read all the regexes into a list
with open(regexes_path, 'r') as f:
    regex_list = f.read().splitlines()

# Open file2 and read all the inputs into another list
with open(inputs_path, 'r') as f:
    input_list = f.read().splitlines()

# Create an empty list to store the results
results = []

# Loop through each regex in the regex list
for regex_index, regex in enumerate(regex_list):
    regex_compiled_path = os.path.join(
        compiled_regexes_path, str(regex_index) + '.txt')
    # Compile the regex using the compiler
    result = subprocess.run([compiler_path, "--regex", regex, "-o",
                            regex_compiled_path, "--emit=compiled", "--binary-format=hex"])
    if result.returncode != 0:
        print('Error compiling regex: ' + regex)
        sys.exit(1)
    for input_str in input_list:
        # Use the `re` module to match the regex with the input
        match = re.search(regex, input_str)
        results.append((regex_compiled_path, input_str, 1 if match else 0))

# Open a output file for writing and write the results into CSV format
with open(results_path, 'w', newline='') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['Regex', 'Input', 'Match'])
    writer.writerows(results)
