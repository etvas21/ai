'''
Created on 2019. 7. 25.

@author: HRKim
'''
import tensorflow as tf
import numpy as np
tf.enable_eager_execution() # 즉시실행

# Data
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]


import matplotlib.pyplot as plt
plt.plot(x_data, y_data, 'o')
plt.ylim(0, 8)

# Hypothesis
v =[1., 2., 3., 4.]
tf.reduce_mean(v) # 2.5
tf.square(3) # 9


# Data
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

# W, b initialize
W = tf.Variable(2.9)
b = tf.Variable(0.5)

hypothesis = W * x_data + b
W.numpy(), b.numpy()
hypothesis.numpy()

plt.plot(x_data, hypothesis.numpy(), 'r-')
plt.plot(x_data, y_data, 'o')
plt.ylim(0, 8)
plt.show()

# Cost
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

with tf.GradientTape() as tape:
    hypothesis = W * x_data + b
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

W_grad, b_grad = tape.gradient(cost, [W, b])
W_grad.numpy(), b_grad.numpy()

# Update parameter
learning_rate = 0.01

# A.assign_sub(B)
#  A = A - B  / A -= B
W.assign_sub(learning_rate * W_grad)
b.assign_sub(learning_rate * b_grad)

#??? W.numpy(), b.numpy()

plt.plot(x_data, hypothesis.numpy(), 'r-')
plt.plot(x_data, y_data, 'o')
plt.ylim(0, 8)

# W, b initialize
W = tf.Variable(2.9)
b = tf.Variable(0.5)


# W, b update
for i in range(100):
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
      print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

plt.plot(x_data, y_data, 'o')
plt.plot(x_data, hypothesis.numpy(), 'r-')
plt.ylim(0, 8)
plt.show()

# predict
print(W * 5 + b)
print(W * 2.5 + b)