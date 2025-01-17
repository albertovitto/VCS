# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
msg = "hello world"
print(msg)


# %%
import numpy as np


# %%
n = 200
fibonacci = np.zeros((n))
fibonacci[0] = 0
fibonacci[1] = 1
index = 2
for index in range(2, n):
    fibonacci[index] = fibonacci[index - 1] + fibonacci[index - 2]


# %%
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show()


# %%

