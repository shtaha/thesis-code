import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


np.random.seed(0)
n = 1000
X = np.zeros((n, 3))
X[:, 0] = np.random.normal(loc=1.0, scale=0.25, size=(n, ))
X[:, 1] = np.random.normal(loc=0.5, scale=0.5, size=(n, ))
X[:, 2] = np.random.normal(loc=0.8, scale=0.3, size=(n, ))

means = X.mean(axis=0)
max_ids = np.argsort(means)[-2:]

print(means, max_ids)

X = pd.DataFrame(X[:, max_ids], columns=max_ids.astype(str))

fig, ax = plt.subplots()
sns.kdeplot(data=X, ax=ax, shade=True)
plt.show()
