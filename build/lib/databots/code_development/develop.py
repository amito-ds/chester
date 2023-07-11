import openai

from databots.brainstorming.task_extends import break_to_subtasks
from databots.code_to_file.code_to_file import write_code_to_file, run_python_file
from databots.utils import MODEL_ENGINE


def developer(text, file_path, temperature=0.99, max_tokens=500):
    pre = "Write python code to:"
    text = pre + text + "\n"
    text = text + "You may or may not use the following instructions:\n" + break_to_subtasks(text) + "\n"
    text = text + "In your next message, write the python code only, the rest as comments #comment" + "\n"
    print("Developer instructions: ", text)
    response = openai.Completion.create(
        engine=MODEL_ENGINE,
        prompt=text,
        temperature=temperature,
        max_tokens=max_tokens
    )

    code_response = response.choices[0].text + "\n"
    print("*********************XXXX******")
    print(code_response)
    print("*********************XXXX******")
    write_code_to_file(code_response, file_path)
    return text, run_python_file(file_path)

