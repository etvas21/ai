'''
Root Mean Square Error(평균제곱근오류)
    여러개의 입력값을 계산할때 임의의 선을 그리고 난후, 이 선이 얼마나 잘 그려졌는지
    평가하여 조금씩 수정해 가는 방법
  MSE가 너무커서  적용하기가 불편한 경우가 있어, MSE의 결과값에 제곱근 처리를 하여
     사용함.
       
    RMSE = SQRT( (1/n ) * ∑((p - y)**2) ) 
    p: 실제값
    y: 예측값
    
    Ref.
    MSE( Mean Square Error ) = (1/n ) * ∑((p - y)**2)     
    
Created on May 21, 2019
@author: hrkim
'''

import numpy as np

# a= 3, b=76으로 임의설정
ab = [ 3,76 ]
data = [ [2,81], [4,93], [ 6,91], [8,97]]

x = [ i[0] for i in data ]
y = [ i[1] for i in data ]


print(x,y)

# y=ax+b에 a,b를 대입하여 결과를 출력하는 함수
def predict(x): 
    return ab[0] * x + ab[1]

# RMSE 값을 구함.
def rmse(p,y):
    return np.sqrt(((p-y)**2).mean())

def rmse_val(predict_result,y ):
    # np.array : ndarray
    # python list: 여러가지 타입의 원소, linked list구현, 메모리 용랼이 크고 속도가 느림, 백터화 연산 불가
    #     a= [[1,2],[3,4]
    #     list: a[0][1]
    # numpy ndarray: 동일타입의 원소,contiguous memory layout, 메모리최적화, 계산속도 향상,백터화 연산 가능
    #     b = np.array(a)
    #     b[0,1] 
    return rmse( np.array(predict_result), np.array(y))

predict_result = []

for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print('공부한 시간 = %.f, 실제점수=%.f, 예측점수=%.f' %(x[i], y[i], predict(x[i])))
    
    
print('RMSE 최종값 :'+str(rmse_val(predict_result,y)))    