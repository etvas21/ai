'''
Created on 2019. 5. 28.
@author: HRKim
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime as dt
import seaborn as sns

col_headers= ["depature_day", "depature_time", "week_day",'driving_time']
col_dtypes = {'depature_day':'str', 'depature_time':'int64'
             , "week_day":'int64', 'driving_time': 'float64'}

df_traffic = pd.read_csv('../data/EX_data_영업소간통행시간_201904_101_105.txt'
                 , sep='|' 
                 , header=None
                 , names= col_headers
                 , dtype= col_dtypes
                 )
             
####################################
# declare
####################################
# HYPORTHESIS
#    w1: 요일
#    w2: 출발시간
#    w3: 기후 (0~1)
#    w4: 공사 (0~1)
#    hypothesis: 소요시간

###
# np.array를 적용을 하지 않으니, 
# hyp = w*x +b 에서
# TypeError: object of type 'RefVariable' has no len() 발생
###
x = np.array(df_traffic[df_traffic['week_day']==1].loc[:,['depature_time']])
y = np.array(df_traffic[df_traffic['week_day']==1].loc[:,['driving_time']])



y = np.array(df_traffic.loc[:,['driving_time']])

w0 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
w1 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
w2 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
w3 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
w4 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
w5 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
w6 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
w7 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
w8 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
w9 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
w10 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
w11 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
w12 = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))

b = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))


#plt.plot(x,y)
#plt.show()

# HYPOTHESIS
hyp =  ( w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6   
    + w7 * x7 + w8 * x8 + w9 * x9 + w10 * x10   
    + w11 * x11 + w12 * x12 
    + b )

# RMSE
rmse = tf.sqrt(tf.reduce_mean(tf.square( hyp - y )))

# learning rate
learning_rate = 0.1

# Gradient Descent
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)
####################################
# learning
####################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(1000):
        sess.run(gradient_descent)
        
        if step % 100 == 0:
              print("Epoch: %.f, RMSE = %.04f,  w = %.4f, b = %.4f"
                    % (step,sess.run(rmse),sess.run(w),sess.run(b)))
    wx = sess.run(w)
    bx = sess.run(b)
    hh = wx * x + bx

with tf.Session() as sess_grp:     
    sess_grp.run(tf.global_variables_initializer())
  
    plt.plot(x,hh)
    
    plt.show()

print('{0:=^50}'.format('End of source'))
print(__file__)