import os
import bz2
import pandas as pd
from io import StringIO


# case_name = "rte_case5_example"
case_name = "l2rpn_2019"
dataset_path = os.path.join(
    os.path.expanduser("~"), "data_grid2op", case_name, "chronics"
)

print(os.listdir(dataset_path))


def read_bz2_to_dataframe(file_path, sep=";"):
    data_csv = bz2.BZ2File(file_path).read().decode()
    return pd.read_csv(StringIO(data_csv), sep=sep)


data = pd.DataFrame()

for chronic in os.listdir(dataset_path):
    chronic_dir = os.path.join(dataset_path, chronic)

    for file in os.listdir(chronic_dir):
        if ("prod_p" in file or "prods_p" in file) and "planned" not in file:
            file_path = os.path.join(chronic_dir, file)
            data_chronic = read_bz2_to_dataframe(file_path, sep=";")
            data = data.append(data_chronic)
            print(file_path)

print(data.max())
print(data.min())
