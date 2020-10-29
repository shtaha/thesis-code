import pandas as pd

freq = ["0.001", "0.11", "FULLFFT", "0.9"]
x = ["0.2", "not float", "0.22", "True"]

df = pd.DataFrame({"freq": freq, "x": x})
print(df)

print(df["freq"].dtype)
print(df["x"].dtype)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(df)
