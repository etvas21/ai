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
# review
####################################
plt.plot(df_traffic['depature_time'],df_traffic['driving_time'])

plt.xlabel('depature time(hour)')
plt.ylabel('driving time(minute)')

plt.show()

####################################
# declare
####################################
# HYPORTHESIS
#    w1: 요일
#    w2: 출발시간
#    w3: 기후 (0~1)
#    w4: 공사 (0~1)
#    hypothesis: 소요시간

x_data = np.array(df_traffic.loc[:,['week_day','depature_time']].astype(float))
y_data = np.array(df_traffic['driving_time'].astype(float))


# random_uniform : 정규분포 난수생성
W = tf.Variable(tf.random_uniform([2,1], 0.0,50.0), dtype='float32')

x_data = np.float32(x_data)

hyp =  tf.matmul(x_data,W)

# RMSE
rmse = tf.sqrt(tf.reduce_mean(tf.square( hyp - y_data )))

# learning rate
learning_rate = 0.1

# Gradient Descent
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)
####################################
# learning
####################################


# 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(500):
        sess.run(gradient_descent)
        
        if step % 100 == 0:
              print("Epoch: %.f, RMSE = %.04f, 기울기 w1 = %.4f, 기울기 w2 = %.4f"
                    % (step,sess.run(rmse),sess.run(W[0]),sess.run(W[1])))
    w0 = sess.run(W[0])
    w1 = sess.run(W[1])
#Epoch: 400, RMSE = 9.3132, 기울기 w1 = 1.7581, 기울기 w2 = 0.2608
#Epoch: 4900, RMSE = 9.3132, 기울기 w1 = 1.7581, 기울기 w2 = 0.2608
#Epoch: 4900, RMSE = 9.3132, 기울기 w1 = 1.9005, 기울기 w2 = 1.0219
wgt = np.array([[w0[0]],[w1[0]]])
hh  = tf.matmul(x_data,wgt)

with tf.Session() as sgrp:     
    sgrp.run(tf.global_variables_initializer())
    
    plt.plot(x_data,sgrp.run(hh))
    
    plt.legend(['A','B'])
    plt.show()


print('{0:=^50}'.format('End of source'))
print(__file__)