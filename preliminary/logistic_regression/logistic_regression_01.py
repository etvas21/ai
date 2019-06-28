'''
Created on 2019. 6. 28.
@author: HRKim
'''

import tensorflow as tf
import numpy as np


# x,y의 데이터 값
x_data = [[1, 2],[2, 3],[3, 1],[4, 3],[5, 3],[6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]       # y 값은 항상 0,1을 가진다. [binary classification]

X = tf.placeholder(tf.float32, shape=[None, 2])     #x_data의 행은 증가할 수 있어 None, 열을 2로 고정
Y = tf.placeholder(tf.float32, shape=[None, 1])

# X의 열이 2임으로 W의 행은 2이고, Y의 값이 1이기 때문에 1==> [2,1]
W = tf.Variable(tf.random_normal([2,1]), name='weight')
# Y의 값이 1이기 때문에 1==> [1]
b = tf.Variable(tf.random_normal([1]), name='bias')

# HYPOTHESIS: y 시그모이드 함수의 방정식을 세움
# y = 1/( 1+ np.e**( W * x_data + b))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# COST FUNCTION: 오차를 구하는 함수
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# LEARNING RATE 학습률 값
learning_rate=0.1

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# tf.cast : Tensor를 새로운 형태로 캐스팅하는데 사용
#   tf.cast( y>0.5, dtype=tf.float64)
#    y의 값이 0.5보다 크면 true임으로 float64 type으로 1. 을 return
#    y의 값이 0.5보다 작으면 true임으로 float64 type으로 0. 을 return
predicted = tf.cast(Y > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


# 학습
print('{0:=^50}'.format('exploration'))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        # run( fetches, feed_dict=None, options=None, run_metadata=None)
        # The value returned by run() has the same shape as the fetches argument,
        # where the leaves are replaced by the corresponding values returned by Tensorflow.
        #a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={X: x_data, Y: y_data})
        cost_val,_= sess.run([cost,train], feed_dict={X:x_data, Y:y_data})
        
        if step % 200 == 0:
            print("step=%d, cost_val=%.4f" % (step , cost_val))

    print('{0:=^50}'.format('Accuracy report'))
    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y):", c, "\nAccuracy: ", a)