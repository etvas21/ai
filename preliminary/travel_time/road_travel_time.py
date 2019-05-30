'''
Created on 2019. 5. 28.
@author: HRKim
'''
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d
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

env.set_matplotlib_font()               # matplotlib를 이용하여 그래프 생성시 한글을 사용할 수 있게함.


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
        plt.plot(np.array(dfx[dfx['week_day']==i_day].loc[:,['depature_time']])
                 , np.array(dfx[dfx['week_day']==i_day].loc[:,['driving_time']])
                 , linewidth = 1.5
                 #, linestyle = linestyles[i_day]
                 , label=day_nm[i_day])
        
    plt.title('[고속도로_경부선]\n서울영업소에서 기흥영업소 까지 요일별 /시간별 평균 소요시간')    
    plt.xlabel('출발시간(시)')
    plt.ylabel('소요시간(분)')
    plt.legend()    
    plt.grid(True)
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
    plt.grid(True)
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
view_graphic_relplot(df_traffic)
df_traffic = load_data(df_traffic)
###

df_traffic_analysis = make_analysis_df(df_traffic)
view_graphic_analysis(df_traffic_analysis)
view_graphic_raw(df_traffic)

####################################
# Linear Regression
####################################  
x_weekday = np.array(df_traffic.loc[:,['week_day']])
x_depature_time = np.array(df_traffic.loc[:,['depature_time']])
y_driving_time = np.array(df_traffic.loc[:,['driving_time']])

w_weekday = tf.Variable(tf.random_uniform([1], 0.0,5.0), dtype='float32')
w_depature_time = tf.Variable(tf.random_uniform([1], 0.0,5.0), dtype='float32')
y_bias = tf.Variable(tf.random_uniform([1], 0.0,5.0), dtype='float32')

hyp = w_weekday * x_weekday + w_depature_time * x_depature_time + y_bias 

# RMSE-Root Means Square Error
rmse = tf.sqrt(tf.reduce_mean(tf.square( hyp - y_driving_time )))

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
              print("Epoch: %.f, RMSE = %.04f, w_weekday = %.4f, w_depature_time = %.4f, y_bias = %.4f"
                    % (step,sess.run(rmse),sess.run(w_weekday),sess.run(w_depature_time),sess.run(y_bias)))

    print("\n\nhyp =%.4f * x_weekday + %.4f * x_depature_time + %.4f"
            % (sess.run(w_weekday),sess.run(w_depature_time),sess.run(y_bias)))
#Epoch: 9999, RMSE = 9.3508, w_weekday = 0.0572, w_depature_time = 1.0576, y_bias = 1.0576
print('{0:=^50}'.format('End of source'))
print(__file__)
#sys.exit(1)
####################################
# declare
####################################
# HYPORTHESIS
#    w1: 요일
#    w2: 출발시간
#    w3: 기후 (0~1)
#    w4: 공사 (0~1)
#    hypothesis: 소요시간
