import os
import zipfile

from lib.data_utils import make_dir


def write_file(file_path, contents):
    make_dir(os.path.dirname(file_path))

    with open(file_path, "w") as f:
        f.write(contents)

    print(f"Created {file_path}")


def read_file(file_path):
    with open(file_path, "r") as f:
        contents = f.read()

    print(f"Reading {file_path}")
    return contents


def zip_directory(directory, zip_name=None, ignore_dirs=("__pycache__",)):
    if not zip_name:
        zip_name = os.path.basename(directory) + ".zip"

    zip_path = os.path.join(directory, zip_name)

    file_paths = []
    for root, directories, files in os.walk(directory):
        if any([ignore_dir in os.path.split(root) for ignore_dir in ignore_dirs]):
            continue

        for filename in files:
            filepath = os.path.join(root, filename)

            if filepath != zip_path:
                file_paths.append(filepath)

    print(f"Creating archive {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in file_paths:
            print(f"\t-- {file}")
            zipf.write(file)


def metadata():
    return "description: this is a non executable code submission."
