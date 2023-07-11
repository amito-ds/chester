import datetime

MODEL_ENGINE = "text-davinci-003"  # or "text-curie-001" for Curie
api_key = "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"


def log(text):  # define a function, `log`
    current_time = datetime.datetime.now()  # set the current time variable to the current time
    print(f'{current_time}: {text}')  # prints the currentTime and the given `text` parameter
