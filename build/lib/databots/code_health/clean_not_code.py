def clean_python_code(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    cleaned_code = []

    for line in content:
        stripped_line = line.strip()
        if stripped_line.startswith('#') or is_python_code(stripped_line):
            cleaned_code.append(line)

    with open(file_path, 'w') as cleaned_file:
        cleaned_file.writelines(cleaned_code)

        # close the file after writing
        cleaned_file.close()


def is_python_code(line):
    try:
        compile(line, '<string>', 'exec')
        return True
    except (SyntaxError, TypeError, ValueError):
        return False
