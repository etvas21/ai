'''
Created on Jul 1, 2019

@author: hrkim
'''
import numpy as np
import tensorflow as tf 

xy = np.loadtxt('../data/data-04-zoo.csv', delimiter=',', dtype=np.float32)

# x,y의 데이터 값
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

nb_classes = 7 #0~6

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None,1])  # 0 ~ 6, shaoe=(?,1)

# if the input indices in rank N, the output will have rank N+1. 
# The new axis is created at dimension axis( default: the new axis is appended at the end )
# [[0],[3] ... ==> [ [ [1 0 0 0 0  0 0] [0 0 0 1 0 0 0] .... 
# ==> [ [1 0 0 0 0 0 0],[0 0 0 1 0 0 0]...
Y_one_hot = tf.one_hot(Y,nb_classes)    # one hot shape = (?,1,7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # shape =(?,7)

W = tf.Variable(tf.random_normal([16,nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)

# cross entropy with logits
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits
                                                 , labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


prediction = tf.argmax(hypothesis,1)    # 확율의 값을  0 ~ 6의 값으로 변환
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step  in range(2001):
        sess.run(optimizer, feed_dict ={X:x_data, Y:y_data})
        if step%200 ==0:
            loss, acc = sess.run([cost,accuracy], feed_dict={X:x_data, Y:y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step,loss,acc))
    
    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict ={X: x_data})
    
    # y_data: (N,1) = flatten ==> (N,) matches pred.shape
    for p,y in zip(pred, y_data.flatten()): # flatten [[1],[2].... ==> [ 1,2 ...]
        print("[{}] prediction: {} True Y: {}".format(p == int(y), p, int(y)))
    