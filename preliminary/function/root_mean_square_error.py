'''
Root Mean Square Error(평균제곱근오류)
Created on May 21, 2019
@author: hrkim
'''

import numpy as np

ab = [ 3,76 ]
data = [ [2,81], [4,93], [ 6,91], [8,97]]

x = [ i[0] for i in data ]
y = [i[1] for i in data ]

print(x,y)

def predict(x): # y=ax+b에 a,b를 대입하여 결과를 출력하는 함수
    return ab[0] * x + ab[1]

def rmse(p,a):
    return np.sqrt(((p-a)**2).mean())

def rmse_val(predict_result,y):
    return rmse(np.array(predict_result), np.array(y))

predict_result = []

for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print('공부한 시간 = %.f, 실제점수=%.f, 예측점수=%.f' %(x[i], y[i], predict(x[i])))
    
    
print('RMSE 최종값 :'+str(rmse_val(predict_result,y)))    