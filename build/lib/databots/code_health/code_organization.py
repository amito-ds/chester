import ast
import io
import os
import sys
from tempfile import NamedTemporaryFile

import black
import openai
import pycodestyle
import pyflakes.api

from databots.code_development.develop import *
from databots.code_development.error_handling import qa_fix_comments
from databots.code_health.clean_not_code import clean_python_code
from databots.utils import *

openai.api_key = api_key


class FormatFile:
    def __init__(self, file_path: str, is_black=True, is_pyflake=True, is_clean_no_code=True):
        self.file_path = file_path
        self.is_black = is_black
        self.is_pyflake = is_pyflake
        self.is_clean_no_code = is_clean_no_code
        self.code_comments = self.get_code_comments()

    def get_code_comments(self):
        report1 = check_file_for_debugger_statements(filepath=self.file_path)
        report2 = get_pyflakes_report(file_path=self.file_path)
        return report1 + "\n" + report2

    def run(self):
        # clean_python_code(self.file_path)
        if self.is_black:
            format_file_with_black(self.file_path)
        # if self.is_pyflake:
        #     qa_fix_comments(comments=self.code_comments, file_path=self.file_path)
        # if self.is_black:
        #     format_file_with_black(self.file_path)


def format_file_with_black(file_path: str) -> None:
    """Formats the file at `file_path` using the Black code formatter."""
    with open(file_path, "r") as f:
        file_contents = f.read()

    # Format the file using Black
    formatted_contents = black.format_file_contents(
        file_contents,
        fast=False,
        mode=black.Mode(),
    )

    # Write the formatted contents back to the file
    with open(file_path, "w") as f:
        f.write(formatted_contents)


def get_pyflakes_report(file_path: str) -> str:
    """Checks the Python file at `file_path` using PyFlakes and returns the report."""
    with open(file_path, 'r') as f:
        tree = compile(f.read(), file_path, 'exec', ast.PyCF_ONLY_AST)

    report = io.StringIO()
    reporter = pyflakes.reporter.Reporter(report, file_path)

    pyflakes.api.check(tree, file_path, reporter)

    return report.getvalue()


def check_file_for_debugger_statements(filepath, pycodestyle_report=
"/Users/amitosi/PycharmProjects/chester/databots/code_health/pycodestyle_report.txt"):
    """Process file using pycodestyle Checker and return all errors."""

    if os.path.isfile(pycodestyle_report):
        with open(pycodestyle_report, "w") as f:
            pass
    else:
        print("File does not exist.")

    if not os.path.isfile(filepath):
        raise ValueError("Invalid file path")

    with open(filepath, "r") as f:
        code = f.read()

    test_file = NamedTemporaryFile(delete=False)
    test_file.write(code.encode())
    test_file.flush()

    # Redirect standard output to a file
    original_stdout = sys.stdout
    sys.stdout = open(pycodestyle_report, "a")

    errors = []
    lines = [line + "\n" for line in code.split("\n")]
    checker = pycodestyle.Checker(filename=test_file.name, lines=lines)

    num_errors = checker.check_all()
    if num_errors == 0:
        os.unlink(test_file.name)
        sys.stdout.close()
        sys.stdout = original_stdout
        return errors

    # Save the printed statements to the input file
    with open(filepath, "w") as f:
        f.write("".join(lines))

    os.unlink(test_file.name)
    sys.stdout.close()
    sys.stdout = original_stdout
    with open(pycodestyle_report, "r") as f:
        file_contents = f.read()
    return file_contents
