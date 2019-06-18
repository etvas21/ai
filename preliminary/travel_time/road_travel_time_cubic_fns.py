'''
3차 다항변수식으로 구현 
Created on 2019. 6.6
@author: HRKim
'''
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D # matplotlib의 2D그래프를 3D 그래프로 그려
from matplotlib import cm, font_manager

# helper module(python)
import datetime as dt
import sys

# helper module(KHR)
import common.sys_env as env

##########################################
# declare variables
##########################################
file_path_traffic = '../data/EX_data_영업소간통행시간_201904_101_105.txt'
day_nm = ['월','화','수','목','금','토','일']

df_traffic =  pd.DataFrame()            # 초기데이터 적재용
df_traffic_analysis =  pd.DataFrame()   # 평균/합계등 통계용 데이터 적재용

env.set_matplotlib_kor()               # matplotlib를 이용하여 그래프 생성시 한글을 사용할 수 있게함.


def load_data(dfx):
    '''
    read data
    '''
    col_headers= ["depature_day", "depature_time", "week_day",'driving_time']
    col_dtypes = {'depature_day':'str', 'depature_time':'int64'
                 , "week_day":'int64', 'driving_time': 'float64'}
        
    dfx = pd.read_csv(file_path_traffic
                 , sep='|' 
                 , header=None
                 , names= col_headers
                 , dtype= col_dtypes
                 )
    
    return dfx
 
def make_analysis_df(dfx):
    '''
    todo : sum, std 등을 추가 
    '''
    #선택한 컬럼을 출력
    #df_traffic[['week_day','depature_time']]
    
    #특정 키워드가 포함되어 있는 레코드필터링
    #검색대상 컬럼이 문자열이어야 함.
    #filter_weekday = ['20190101','20190102']
    #for weekday in filter_weekday:
    #    rslt = df_traffic[df_traffic['depature_day'].str.contains(weekday)]
    
    #df_traffic['depature_day'].str.replace('2019','')
    
    #df_traffic.groupby(['week_day','depature_time']).sum()    
    #df_traffic.groupby(['week_day','depature_time']).count()
    #df_traffic.groupby(['week_day','depature_time']).mean()
    #df_traffic.groupby(['week_day','depature_time']).agg(['sum','mean'])
        
    # ...
    dfg = dfx.reset_index().groupby(['week_day','depature_time'],as_index=False).mean()
      
    return dfg

def view_graphic_analysis(dfx):
    '''
        평균을 이용하여 그래프 그리기 
    '''
    linestyles = [ '-', '--', '-.', ':',':','--','--']
    for i_day in range(7):
        plt.grid(True)
        plt.plot(np.array(dfx[dfx['week_day']==i_day].loc[:,['depature_time']])
                 , np.array(dfx[dfx['week_day']==i_day].loc[:,['driving_time']])
                 , linewidth = 1.5
                 #, linestyle = linestyles[i_day]
                 , label=day_nm[i_day])
        
    plt.title('[고속도로_경부선]\n서울영업소에서 기흥영업소 까지 요일별 /시간별 평균 소요시간')    
    plt.xlabel('출발시간(시)')
    plt.ylabel('소요시간(분)')
    plt.legend()    
    
    #plt.show()    # 기본데이터와 한개의 그래프에서 보기 위하여 실행 않음.

def view_graphic_relplot(dfx):

    sns.relplot(x='depature_time',y='driving_time'
                , col = 'week_day'  # subgraphic으로 나누는 기준
                , col_wrap = 4
                , style = 'week_day'
                , hue ='week_day'
                , kind = 'line'
                , dashes = False
                , markers = True
                , data = dfx)
    plt.tight_layout()
        
    plt.show()
    
def view_graphic_raw(dfx):
    x_depature_time = np.array(dfx.loc[:,['depature_time']])
    y_driving_time = np.array(dfx.loc[:,['driving_time']])
        
    plt.plot(x_depature_time,y_driving_time, linewidth = 0.3)
    
    plt.show()


def hyp_linear_polynomial(dfx):
    dfx = dfx.loc[(df_traffic['week_day'] == 5)]
    y_driving_time = np.array(df_traffic.loc[:,['driving_time']])
    
    for i in range(24):
        dfx['x'+str(i)] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == i else 0 , axis = 1 )
    '''        
    dfx['x0'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 0 else 0 , axis = 1 )
    dfx['x1'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 1 else 0 , axis = 1 )
    dfx['x2'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 2 else 0 , axis = 1 )
    dfx['x3'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 3 else 0 , axis = 1 )
    dfx['x4'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 4 else 0 , axis = 1 )
    dfx['x5'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 5 else 0 , axis = 1 )
    dfx['x6'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 6 else 0 , axis = 1 )
    dfx['x7'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 7 else 0 , axis = 1 )
    dfx['x8'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 8 else 0 , axis = 1 )
    dfx['x9'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 9 else 0 , axis = 1 )
    dfx['x10'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 10 else 0 , axis = 1 )
    dfx['x11'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 11 else 0 , axis = 1 )
    dfx['x12'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 12 else 0 , axis = 1 )
    dfx['x13'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 13 else 0 , axis = 1 )
    dfx['x14'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 14 else 0 , axis = 1 )
    dfx['x15'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 15 else 0 , axis = 1 )
    dfx['x16'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 16 else 0 , axis = 1 )
    dfx['x17'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 17 else 0 , axis = 1 )
    dfx['x18'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 18 else 0 , axis = 1 )
    dfx['x19'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 19 else 0 , axis = 1 )
    dfx['x20'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 20 else 0 , axis = 1 )
    dfx['x21'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 21 else 0 , axis = 1 )
    dfx['x22'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 22 else 0 , axis = 1 )
    dfx['x23'] = dfx.apply(lambda x: x['driving_time'] if x['depature_time'] == 23 else 0 , axis = 1 )
    '''
    x0  = np.array(dfx.loc[:,['x0']]) 
    x1  = np.array(dfx.loc[:,['x1']]) 
    x2  = np.array(dfx.loc[:,['x2']]) 
    x3  = np.array(dfx.loc[:,['x3']]) 
    x4  = np.array(dfx.loc[:,['x4']]) 
    x5  = np.array(dfx.loc[:,['x5']]) 
    x6  = np.array(dfx.loc[:,['x6']]) 
    x7  = np.array(dfx.loc[:,['x7']]) 
    x8  = np.array(dfx.loc[:,['x8']]) 
    x9  = np.array(dfx.loc[:,['x9']]) 
    x10 = np.array(dfx.loc[:,['x10']])
    x11 = np.array(dfx.loc[:,['x11']])
    x12 = np.array(dfx.loc[:,['x12']])
    x13 = np.array(dfx.loc[:,['x13']])
    x14 = np.array(dfx.loc[:,['x14']])
    x15 = np.array(dfx.loc[:,['x15']])
    x16 = np.array(dfx.loc[:,['x16']])
    x17 = np.array(dfx.loc[:,['x17']])
    x18 = np.array(dfx.loc[:,['x18']])
    x19 = np.array(dfx.loc[:,['x19']])
    x20 = np.array(dfx.loc[:,['x20']])
    x21 = np.array(dfx.loc[:,['x21']])
    x22 = np.array(dfx.loc[:,['x22']])
    x23 = np.array(dfx.loc[:,['x23']])

    w0 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')    
    w1 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w2 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w3 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w4 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w5 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w6 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w7  = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w8  = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w9  = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w10 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w11 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w12 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w13 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w14 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w15 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w16 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w17 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w18 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w19 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w20 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w21 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w22 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    w23 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
    
    b = tf.Variable(tf.random_uniform([1], -50.0,50.0), dtype='float32')    
    
    hyp = b + w0*x0 + w1*x1+ w2*x2+ w3*x3+ w4*x4+ w5*x5+ w6*x6 + w7*x7+ w8*x8+ w9*x9+ w10*x10 + w11*x11 + w12*x12+ w13*x13 + w14*x14 + w15*x15 + w16*x16 + w17*x17 + w18*x18+ w19*x19 + w20*x20 + w21*x21 + w22*x22 + w23*x23 
    
    # RMSE-Root Means Square Error
    rmse = tf.sqrt(tf.reduce_mean(tf.square( hyp - y_driving_time )))

    # learning rate
    learning_rate = 0.01
    
    # Gradient Descent
    gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

    ####################################
    # learning
    ####################################
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for step in range(200):
            sess.run(gradient_descent)
            
            if step % 100 == 0:
                  print("Epoch: %.f, RMSE = %.04f, w1 = %.4f, w2 = %.4f, w3 = %.4f, b = %.4f"
                        % (step,sess.run(rmse),sess.run(w1),sess.run(w2),sess.run(w3),sess.run(b)))
    
####################################
# main
####################################  
df_traffic = load_data(df_traffic)
hyp_linear_polynomial(df_traffic)
sys.exit(1)
###
#view_graphic_relplot(df_traffic)
#df_traffic = load_data(df_traffic)
###

#df_traffic_analysis = make_analysis_df(df_traffic)
#view_graphic_analysis(df_traffic_analysis)
#view_graphic_raw(df_traffic)

####################################
# Linear Regression
####################################
df_traffic = df_traffic.loc[(df_traffic['week_day'] == 5)]  
x_depature_time = np.array(df_traffic.loc[:,['depature_time']])
y_driving_time = np.array(df_traffic.loc[:,['driving_time']])


w1 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
w2 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
w3 = tf.Variable(tf.random_uniform([1], -5.0,5.0), dtype='float32')
b = tf.Variable(tf.random_uniform([1], -50.0,50.0), dtype='float32')
#hyp = w_weekday * x_weekday + w_depature_time * x_depature_time + y_bias
#hyp = ((w_weekday*x_weekday-x_bias)**3)*((w_depature_time*x_depature_time-y_bias)**3) 
hyp = w1*x_depature_time**3 + w2*x_depature_time**2 + w3*x_depature_time + b

# RMSE-Root Means Square Error
rmse = tf.sqrt(tf.reduce_mean(tf.square( hyp - y_driving_time )))

# learning rate
learning_rate = 0.01

# Gradient Descent
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)
####################################
# learning
####################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(gradient_descent)
        
        if step % 100 == 0:
              print("Epoch: %.f, RMSE = %.04f, w1 = %.4f, w2 = %.4f, w3 = %.4f, b = %.4f"
                    % (step,sess.run(rmse),sess.run(w1),sess.run(w2),sess.run(w3),sess.run(b)))

    #print("\n\nhyp =%.4f * x_weekday + %.4f * x_depature_time + %.4f = %f w/3,9"
    #        % (sess.run(w_weekday),sess.run(w_depature_time),sess.run(y_bias)
    #        , sess.run(w_weekday)*3+sess.run(w_depature_time)*6 + sess.run(y_bias)))  # 26.474686
    
#print(df_traffic.loc[(df_traffic['week_day'] == 3)
#                            & (df_traffic['depature_time'] == 6)])   

print('{0:=^50}'.format('End of source'))
print(__file__)