import openai

from databots.code_to_file.code_to_file import write_code_to_file, run_python_file
from databots.utils import MODEL_ENGINE


def qa(original_task, error, file_path, temperature=0.99, max_tokens=500):
    conversation = f"I got a task: {original_task}\n"
    conversation = conversation + "This is my code:\n"
    code = open(file_path).read() + "\n"
    conversation = conversation + code + "\n"
    conversation = conversation + "And error: \n" + error + "\n"
    conversation = conversation + "Fix the code given this info"

    response = openai.Completion.create(
        engine=MODEL_ENGINE,
        prompt=conversation,
        temperature=temperature,
        max_tokens=max_tokens
    )
    print(conversation)
    qa_response = response.choices[0].text + "\n"
    write_code_to_file(qa_response, file_path)
    return qa_response, run_python_file(file_path)


def qa_fix_comments(comments, file_path, temperature=0.99, max_tokens=500):
    conversation = "Given this code:\n"
    code = open(file_path).read() + "\n"
    conversation = conversation + code + "\n"
    conversation = conversation + "And Comments: \n" + comments + "\n"
    conversation = conversation + "Fix the code given this info"

    response = openai.Completion.create(
        engine=MODEL_ENGINE,
        prompt=conversation,
        temperature=temperature,
        max_tokens=max_tokens
    )
    print(conversation)
    qa_response = response.choices[0].text + "\n"
    write_code_to_file(qa_response, file_path)
    return qa_response, run_python_file(file_path)

# TODO: pip install
