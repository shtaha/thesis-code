import os
import bz2
import pandas as pd
from io import StringIO


case_name = "rte_case5_example"
dataset_path = os.path.join(
    os.path.expanduser("~"), "data_grid2op", case_name, "chronics"
)

print(os.listdir(dataset_path))


def read_bz2_to_dataframe(file_path, sep=";"):
    data_csv = bz2.BZ2File(file_path).read().decode()
    return pd.read_csv(StringIO(data_csv), sep=sep)


for chronic in os.listdir(dataset_path):
    chronic_dir = os.path.join(dataset_path, chronic)

    for file in os.listdir(chronic_dir):
        file_path = os.path.join(chronic_dir, file)
        data = read_bz2_to_dataframe(file_path, sep=";")
        # print(data.to_string())

        print(file)
        print(data)
        print("\n")
    break
