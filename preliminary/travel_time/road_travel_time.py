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
import sys
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm, font_manager
# for font
import common.sys_env as env

day_nm = ['MON','TUE','WED','THU','FRI','SAT','SUN']

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
# 선택한 컬럼을 출력
#print(df_traffic[['week_day','depature_time']])

# 특정 키워드가 포함되어 있는 레코드필터링
# 검색대상 컬럼이 문자열이어야 함.
#weekdayss = ['20190101','20190102']
#for weekday in weekdayss:
#    rslt = df_traffic[df_traffic['depature_day'].str.contains(weekday)]
#print(rslt.count())

#print(df_traffic.groupby(['week_day','depature_time']).sum())    
#print(df_traffic.groupby(['week_day','depature_time']).count())

#print(df_traffic['depature_day'].str.replace('2019',''))

#print(df_traffic.groupby(['week_day','depature_time']).mean())
dfx = df_traffic.reset_index().groupby(['week_day','depature_time'],as_index=False).mean()

print(dfx)

sys.exit(1)
####################################
# declare
####################################
# HYPORTHESIS
#    w1: 요일
#    w2: 출발시간
#    w3: 기후 (0~1)
#    w4: 공사 (0~1)
#    hypothesis: 소요시간


x_weekday = np.array(df_traffic.loc[:,['week_day']])
x_depature_time = np.array(df_traffic.loc[:,['depature_time']])
y_driving_time = np.array(df_traffic.loc[:,['driving_time']])

##########################################################
#yy = 1.1002*xx[0] + 0.3198*xx[1]

env.set_matplotlib_font()

plt.plot(x_depature_time,y_driving_time)

plt.xlabel('출발시간')
plt.ylabel('주행시간')

plt.show()

##########################################################

w_weekday = tf.Variable(tf.random_uniform([1], 0.0,10.0), dtype='float32')
w_depature_time = tf.Variable(tf.random_uniform([1], 0.0,30.0), dtype='float32')

hyp = w_weekday * x_weekday + w_depature_time * x_depature_time 

# RMSE
rmse = tf.sqrt(tf.reduce_mean(tf.square( hyp - y_driving_time )))


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
                    % (step,sess.run(rmse),sess.run(w_weekday),sess.run(w_depature_time)))



#sys.exit(1)
print('{0:=^50}'.format('End of source'))
print(__file__)