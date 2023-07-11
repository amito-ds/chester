import os

import openai

from databots.code_development.error_handling import qa
from databots.code_health.code_organization import FormatFile
from databots.code_to_file.code_to_file import *
from databots.modify_code.code_modification import modify_code
from databots.utils import *

openai.api_key = api_key


def modify(text,
           file_prefix="/Users/amitosi/PycharmProjects/chester/databots",
           python_file="generated_code.py"):
    conversation_collector = []
    log(f"Original task: {text}")
    file_path = os.path.join(file_prefix, python_file)

    # modify code
    conversation, error = modify_code(text, file_path)
    conversation_collector.append(conversation)

    # Fix errors
    count_fixes = 1
    while error is not None:
        log("Found bugs in the code! Fixing...")
        conversation, error = qa(original_task=text, file_path=file_path, error=error,
                                 max_tokens=700)
        conversation_collector.append(conversation)
        count_fixes += 1
        if count_fixes > 7:
            print("Something is wrong, try re-submit the task please")
            break

    log("Code Ran With No Errors!")

    # Run code
    log("Code outputs:")
    execute_code(file_path)

    # Organize code
    log("Organizing Code Style")
    FormatFile(file_path=file_path).run()
    return conversation_collector
