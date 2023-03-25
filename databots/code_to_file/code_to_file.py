import subprocess


def run_python_file(path: str):
    # will return None if not error, otherwise the error
    try:
        subprocess.run(['python', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return None
    except subprocess.CalledProcessError as e:
        return e.stderr.decode()


def execute_code(path: str):
    # run the code
    result = subprocess.run(['python', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    print(result.stdout.decode())


def write_code_to_file(code: str, path: str):
    with open(path, 'w') as file:
        file.write(code)
