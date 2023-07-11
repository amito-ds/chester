import openai

from databots.utils import *

openai.api_key = api_key


def review_code(file_path, text="", temperature=0.99, max_tokens=200):
    conversation = text + "\nThis is the code:\n"
    code = open(file_path).read() + "\n"
    conversation = conversation + code + "\n"

    conversation = conversation + "Review the code, code quality, code style." \
                                  "Suggest ideas for improvements. " \
                                  "Give practical advices"

    response = openai.Completion.create(
        engine=MODEL_ENGINE,
        prompt=conversation,
        temperature=temperature,
        max_tokens=max_tokens
    )

    review_response = response.choices[0].text + "\n"
    print(conversation)
    print("******* Answer:\n")
    print(review_response)

    return review_response
