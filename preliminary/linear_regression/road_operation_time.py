'''
Created on 2019. 5. 27.

@author: HRKim
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime as dt


col_headers= ["depature_day", "depature_time", "depature_office", "arrival_office", "car_type", "lead_time",'unused']
col_dtypes = {'depature_day':'str', 'depature_time':'int64'
             , "depature_office":'int64',"arrival_office":'int64'
             , "car_type":'int64', "lead_time":'float64', 'unused': 'str'}

df_traffic = pd.read_csv('../data/EX_data_영업소간통행시간_201904_101_105.txt'
                 , sep='|' 
                 , header=None
                 , names= col_headers
                 , dtype= col_dtypes
                 )
                
dfx = df_traffic.assign(week_day = pd.to_datetime(df_traffic['depature_day']).dt.weekday)
dfx['lead_time'] = dfx['lead_time'] / 60
dfx.drop(['depature_office','arrival_office','car_type','unused'],axis=1,inplace=True)

dfx = dfx.loc[(dfx['week_day'] == 1)]     
print(dfx)
'''                                                   
# print(df_traffic.columns)
# car type
#    승용자(1종), 버스(2종), 화물차(3~12종)
# 기흥:105, 서울:101
# depature_office를 str으로 read를 하고 filter를 '101'로 하면 filter가 되지 않아서
# depature_office를 int64로 지정을 하고 처리하여 정상적으로 처리됨
# 아마도, str로 읽으면 앞뒤에 space가 있는것 같음.
#dfx = df_traffic.loc[df_traffic.car_type == '1']

# Data Analysis
1. read file
2. Check shape, head
3. Extract only the column need
4. Check head/tail
5. Eye check data type of each columns
6. Check Data range
7. Check shape after eliminate value of 'NaN'
'''

def check_head():
    # 2. Check shape, head
    print('{0:=^50}'.format('Analysis shape'))
    print(df_traffic.shape)
    
def check_head_tail():
    # 처음 5줄 보기
    print('{0:=^50}'.format('Analysis Data (head)'))  
    print(df_traffic.head(5))
    print('{0:=^50}'.format('Analysis Data (tail)'))
    print(df_traffic.tail(5))

def check_info():
    # data 정보 확인
    print(df_traffic.info())

    # 각 정보별 특징을 더자세히( count, mean, std, max ..)
    print(df_traffic.describe())

def append_weekend():
    print('{0:=^50}'.format('convert week_day'))
    
    for i in range(len(dfx)):
        dfx.iloc[i,7] = dt.datetime(int(dfx.iloc[i]['depature_day'][0:4])
                                                    , int(dfx.iloc[i]['depature_day'][4:6])
                                                    , int(dfx.iloc[i]['depature_day'][6:8])).weekday()
####################################
# learning
####################################


####################################
# applying
####################################
'''
check_head()
check_head_tail()
check_info()
'''
print(dfx.info())        

#plt.plot(dfx['depature_time'],dfx['lead_time'])

#plt.xlabel('depature time(hour)')
#plt.ylabel('lead time(minute)')

#plt.show()

# HYPORTHESIS
#    w1: 요일
#    w2: 출발시간
#    w3: 기후 (0~1)
#    w4: 공사 (0~1)
#    hypothesis: 소요시간

x1 = np.array(dfx['week_day'])
x2 = np.array(dfx['depature_time'])
h = np.array(dfx['lead_time'])

w1 = tf.Variable(tf.random_uniform([1], 0,20, dtype=tf.float64, seed=0))
w2 = tf.Variable(tf.random_uniform([1], 0,50, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0,10, dtype=tf.float64, seed=0))


print( 'x1 =' , type(x1), ' x2 = ', type(x2), '\nh = ', type(h), type(w1), type(w2))

hyp =  w1 * x1 + w2 * x2 + b


# RMSE
rmse = tf.sqrt(tf.reduce_mean(tf.square( hyp - h )))

# learning rate
learning_rate = 0.1

# Gradient Descent
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)


# 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        sess.run(gradient_descent)
        
        if step % 100 == 0:
              print("Epoch: %.f, RMSE = %.04f, 기울기 w1 = %.4f, 기울기 w2 = %.4f, y절편 b = %.4f" 
                    % (step,sess.run(rmse),sess.run(w1),sess.run(w2),sess.run(b)))

    w1 = 49
    w2 = 43
    b = 202
    hyp = 9.2329 * x1 -0.2680 * x2 + 1.1639
    plt.plot(x1,x2,hyp)
    plt.show()


print('{0:=^50}'.format('End of source'))
print(__file__)