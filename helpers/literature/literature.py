import matplotlib.pyplot as plt
import numpy as np

i_max = 1
i = i_max * np.linspace(0, 1.1, 1000)
m = np.maximum(0, 1 - i / i_max)

r = 1 - np.square((1 - m))
r_linear = m
r_sqrt = np.sqrt(m)
r_cubic = 1 - np.power(1 - m, 3)

plt.figure(figsize=(16, 9))
plt.plot(i, m, label="margin")
plt.legend()
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(i, r, label="baseline")
plt.plot(i, r_linear, label="linear")
plt.plot(i, r_sqrt, label="sqrt")
plt.plot(i, r_cubic, label="cubic")
plt.legend()
plt.show()
