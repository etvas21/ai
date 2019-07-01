'''
Created on 2019. 6. 28.

@author: HRKim
'''
import tensorflow as tf

x_data = [1,2,3]
y_data = [ 1,2,3]

W = tf.Variable(tf.random_normal([1]),name ='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

for step in range(21):
    sess.run(update, feed_dict={X:x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X:x_data, Y: y_data}), sess.run(W))