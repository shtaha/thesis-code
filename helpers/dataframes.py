import pandas as pd

df1 = pd.DataFrame({
    "id": [0, 1, 8, 3],
    "a": ["a0", "a1", "a2", "a3"],
    "b": ["b0", "b1", "b2", "b3"],
    "c": ["c0", "c1", "c2", "c3"],
    "trafo": [False, False, False, False],
})

df2 = pd.DataFrame({
    "id": [0, 1, 2, 3],
    "a": ["ta4", "ta5", "ta6", "ta7"],
    "b": ["tb4", "tb5", "tb6", "tb7"],
    "c": ["tc4", "tc5", "tc6", "tc7"],
    "trafo": [True, True, True, True],
})

df1.set_index("id", inplace=True)
df2.set_index("id", inplace=True)

print(df1.to_string())
print(df2.to_string())

df = df1.append(df2, ignore_index=False)
print(df.to_string())
df.reset_index(inplace=True)
print(df.to_string())
