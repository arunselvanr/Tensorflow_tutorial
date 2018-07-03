import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)[:, np.newaxis] #[:, np.newaxis] takes an array (n, ) and converts it into (n,1)
noise = np.random.normal(0.0, .5, size=x.shape)
y = np.power(x, 2) + noise

plt.scatter(x, y, c='red')
plt.show()