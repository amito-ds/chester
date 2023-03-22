import os

source_root = os.getcwd()  # Get the current working directory as the source root
folders = []

# Recursively search for all sub-folders under the source root
for root, dirnames, filenames in os.walk(source_root):
    for dirname in dirnames:
        folders.append(os.path.join(root, dirname))

print(folders)
