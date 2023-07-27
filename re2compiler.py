import os
import subprocess
import argparse

compiler_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'build', 'ciceroc')


def compile(inputfile=None, data=None, o=None,
            dotast=None, dotir=None, dotcode=None,
            O1=None, no_postfix=None, no_prefix=None,
            backend=None, frontend=None):
    if backend is not None or frontend is not None:
        print("Error: backend and frontend must be None, unsupported")
        quit(1)

    if no_postfix is not None or no_prefix is not None:
        print("Error: no_postfix and no_prefix must be None, unsupported")
        quit(1)

    if inputfile:
        if data:
            print("Error: inputfile and data cannot be used together")
            quit(1)
        with open(inputfile, "r") as f:
            data = f.read()
    elif data is None:
        data = input('Enter your regex: ')
    if dotast is not None or dotir is not None or dotcode is not None:
        print("Error: dotast, dotir and dotcode must be None, unsupported")
        quit(1)

    command = [compiler_path, "--regex", data,
               "--emit=compiled", "--binary-format=hex", "-o", "-"]

    if O1:
        command += ["-Oall"]

    # Run command
    result = subprocess.run(command, stdout=subprocess.PIPE)

    command_output = result.stdout.decode('utf-8')

    if result.returncode != 0:
        print('Error compiling regex: ' + command_output)
        quit(1)

    if o is not None:
        with open(o, "w") as f:
            f.write(command_output)

    return command_output


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='compile a regular expression into code that can be executed by re2coprocessor(https://github.com/necst/cicero).')
    arg_parser.add_argument(
        'inputfile', type=str, help='input file containing the regular expression.', default=None, nargs='?')
    arg_parser.add_argument(
        '-data', type=str, help='allows to pass the input string representing the regular expression directly via parameter .', default=None, nargs='?')
    arg_parser.add_argument(
        '-o', type=str, help='output file containing the code that represent the regular expression.', default='a.out', nargs='?')
    arg_parser.add_argument(
        '-O1', 			help='perform simple optimizations', default=False, action='store_true')

    args = arg_parser.parse_args()

    compile(**args.__dict__)
