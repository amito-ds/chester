from termcolor import colored


def print_step_message(step_name: str, color: str = "blue"):
    print(colored("\n#### Starting {} step ####\n".format(step_name), color))
