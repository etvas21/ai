'''
    Gradient Descent
    
Created on 2019. 5. 21.
@author: HRKim
'''

import tensorflow as tf

# x,y
data = [ [2,81], [4,93], [ 6,91], [8,97]]
x_data = [ x_row[0] for x_row in data ]
y_data = [ y_row[1] for y_row in data ]

# 기욹기 a, 절편 b의 값을 잉믜로 정함.
# 단, 기물기  0 ~ 10 , 절편 0 ~ 100  범위에서 변하게 함.
# tf.random_uniform(shape, minval, maxval, dtype, seed, name )
a = tf.Variable(tf.random_uniform([1], 0,10, dtype = tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0,100, dtype = tf.float64, seed=0))

# y에 대한 일차방정시 정의 ( 가설 Hypothesis )
y = a * x_data + b

# tensorflow의 RMSE 함수
rmse = tf.sqrt( tf.reduce_mean( tf.square( y - y_data)))


# 학습률 값 
learning_rate = 0.1

# RMSE 값을 최소로하는 값 찾기.
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# Tensorflow를 이용한 학습

# Session 객체 생성
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())
    
    # 2001번 실행(epoch)
    for step in range(2001):
        sess.run(gradient_descent)
        
        if step % 100 == 0:
            print("Epoch: %.f , RMSE= %.04f, weight a = %.4f, bias b = %.4f" 
                        %(step, sess.run(rmse),sess.run(a), sess.run(b)))
print('{0:=^50}'.format('End of source'))            
    
