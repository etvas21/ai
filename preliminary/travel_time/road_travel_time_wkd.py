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

w = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))
b = tf.Variable(tf.random_uniform([1], -10,10, dtype=tf.float32, seed=0))

print(x)
print(y)
print( 'x =' , type(x), '\ny = ', type(y), '\nw = ', type(w), '\nb = ' ,type(b))
print( 'x =' , x.shape, '\ny = ', y.shape, '\nw = ', w.shape, '\nb = ' ,b.shape)

plt.plot(x,y)
plt.show()

# HYPOTHESIS
hyp =  w * x + b

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
    
    for step in range(5000):
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