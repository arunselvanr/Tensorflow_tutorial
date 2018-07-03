import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)

y_relu = tf.nn.relu(x)
y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)

with tf.Session() as sess:
    y_relu, y_sigmoid, y_tanh, y_softplus = sess.run([y_relu, y_sigmoid, y_tanh, y_softplus])

plt.figure(1, figsize=(8,6))

plt.subplot(221) #Create a 2x2 grid, then place it in the first position
plt.plot(x, y_relu, c='red', label='relu')
plt.ylim((y_relu[0]-1, y_relu[-1]+1))
plt.legend(loc='best') #Choose the best location to place the label in the plot

plt.subplot(222)
plt.plot(x, y_sigmoid, c='orange', label='sigmoid')
plt.ylim((y_sigmoid[0]-1, y_sigmoid[-1]+1))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x, y_tanh, c='blue', label='tanh')
plt.ylim((y_tanh[0]-1, y_tanh[-1]+1))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x, y_softplus, c='green', label='softplus')
plt.ylim((y_softplus[0]-1, y_softplus[-1]+1))
plt.legend(loc='best')

plt.show()
