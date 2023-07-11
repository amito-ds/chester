import openai

from databots.utils import MODEL_ENGINE, api_key

openai.api_key = api_key


# todo: handle this
def break_to_subtasks(text):
    temperature = 0.99
    max_tokens = 500
    pre = "The overall task is to write python code "
    text = pre + text
    text = text + "\nDo a code planning. Break it down to subtasks, it will go directly to developer to implement." \
                  "\nif you say to use some model, specify where you get the model first." \
                  "Be clear that the answer should contain python code only."

    response = openai.Completion.create(
        engine=MODEL_ENGINE,
        prompt=text,
        temperature=temperature,
        max_tokens=max_tokens
    )

    code_subtasks = response.choices[0].text + "\n"
    return code_subtasks


def brainstorm_idea(text):
    temperature = 0.99
    max_tokens = 500
    pre = "The overall task is to write python code \n"
    text = pre + text
    text = text + """
    Simulate brainstorm to solve implement this idea with python\n 
    Discuss the purpose and goal of the ideam\n 
    Discuss what would get implemented \n 
    Address concerns\n 
    Discuss potential issues and propose solutions\n 
    Topic 5: Wrap up \n 
    Summarize key points and decisions\n 
    """

    response = openai.Completion.create(
        engine=MODEL_ENGINE,
        prompt=text,
        temperature=temperature,
        max_tokens=max_tokens
    )

    code_subtasks = response.choices[0].text + "\n"
    return code_subtasks
