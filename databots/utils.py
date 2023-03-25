import datetime

MODEL_ENGINE = "text-davinci-003"  # or "text-curie-001" for Curie


def log(text):  # define a function, `log`
    current_time = datetime.datetime.now()  # set the current time variable to the current time
    print(f'{current_time}: {text}')  # prints the currentTime and the given `text` parameter
