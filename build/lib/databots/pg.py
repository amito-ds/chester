from databots.code_health.clean_not_code import clean_python_code
from databots.code_health.code_organization import FormatFile
from databots.from_scratch import create_from_scratch
from databots.modify_file import modify

# initiate
file = "games.py"
file_path = "/Users/amitosi/PycharmProjects/chester/databots/games.py"

text = """
Write a function to clean text. include 3 tests
"""
# create_from_scratch(text, python_file=file)
#
# # modify
# text = """
# Add comments and docstring to the code
# """
# # modify(text=text, python_file=file)
#
# # review
# file = "/Users/amitosi/PycharmProjects/chester/chester/run/full_run.py"
# review_code(file_path=file)
#
text = """
fix bugs
"""
# FormatFile(file_path=file_path).run()
# modify(text=text, python_file=file)
#

# clean_python_code(file_path)
