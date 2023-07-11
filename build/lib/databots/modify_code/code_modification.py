import openai

from databots.code_to_file.code_to_file import write_code_to_file, run_python_file
from databots.utils import MODEL_ENGINE, api_key

openai.api_key = api_key


def modify_code(text, file_path, temperature=0.99, max_tokens=500):
    conversation = "This is my code:\n"
    code = open(file_path).read() + "\n"
    conversation = conversation + code + "\n"
    conversation = conversation + "Modify the code to do the following: \n" + text + "\n"
    conversation = conversation + "In your answer, re-write the whole file, and keep the same logic"

    response = openai.Completion.create(
        engine=MODEL_ENGINE,
        prompt=conversation,
        temperature=temperature,
        max_tokens=max_tokens
    )
    print(conversation)
    response = response.choices[0].text + "\n"
    write_code_to_file(response, file_path)
    return response, run_python_file(file_path)
