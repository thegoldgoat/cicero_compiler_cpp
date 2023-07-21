import csv
import os
import re
import sys

if len(sys.argv) != 2:
    print('Usage: python3 generate_csv.py <output_file>')
    sys.exit(1)


# Get the absolute path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

regexes_path = os.path.join(script_dir, 'regexes.txt')
inputs_path = os.path.join(script_dir, 'inputs.txt')
results_path = sys.argv[1]

# Open file1 and read all the regexes into a list
with open(regexes_path, 'r') as f:
    regex_list = f.read().splitlines()

# Open file2 and read all the inputs into another list
with open(inputs_path, 'r') as f:
    input_list = f.read().splitlines()

# Create an empty list to store the results
results = []

# Loop through each regex in the regex list
for regex in regex_list:
    for input_str in input_list:
        # Use the `re` module to match the regex with the input
        match = re.search(regex, input_str)
        results.append((regex, input_str, 1 if match else 0))

# Open a output file for writing and write the results into CSV format
with open(results_path, 'w', newline='') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['Regex', 'Input', 'Match'])
    writer.writerows(results)