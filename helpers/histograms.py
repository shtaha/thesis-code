import matplotlib.pyplot as plt
import numpy as np

n = 10000
n_bins = 25
# obj_dn = np.abs(1.0 - np.random.chisquare(df=2, size=(n, )))
obj_dn = np.random.beta(a=2.0, b=0.25, size=(n, ))
obj = np.ones_like(obj_dn)
rel_gain = np.divide(obj_dn, obj + 1e-9)

fig, ax = plt.subplots()
plt.hist(rel_gain, bins=n_bins, histtype="step")
plt.xlabel("Relative")
plt.show()
