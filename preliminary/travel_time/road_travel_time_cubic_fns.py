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


####################################
# main
####################################  
df_traffic = load_data(df_traffic)

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