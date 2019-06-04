'''
Created on 2019. 6. 4.
@author: HRKim
[출처:기초수학으로 이해사는 머신러닝 알고리즘]
'''
import numpy as np
import matplotlib.pyplot as plt
import sys

# Load Training data
train_data_file = '../data/click.csv'
train_data = np.loadtxt(train_data_file, delimiter=',', skiprows=1)

train_x = train_data[:,0]
train_y = train_data[:,1]

theta0 = np.random.rand()
theta1 = np.random.rand()

# 표준화
mu = train_x.mean()
sigma = train_x.std()

def f(x):
    # Hypothesis
    return theta0 + theta1 * x

def E(x,y):
    # cost function
    return 0.5 * np.sum((y-f(x))**2)

def standardize(x):
    return ( x - mu )/sigma

train_z = standardize(train_x)

ETA = 1e-3      # 학습률

diff = 1        #오차의차분

count = 0       #갱신횟수

#learning
error = E(train_z, train_y)

while diff > 1e-2:
    #갱신 결과를 임시변수에 저장
    tmp0 = theta0 - ETA * np.sum(f(train_z)-train_y)
    tmp1 = theta1 - ETA * np.sum(((f(train_z)-train_y)*train_z))
    
    #매개변수 갱신
    theta0 = tmp0
    theta1 = tmp1
    
    #이전회의 오차와의 차분 계산
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error
    #로그 출력
    count += 1
    log = '{}회째: theta0 ={:.3f}, theta1 ={:.3f}, 차분={:.4f}'
    print(log.format(count,theta0,theta1,diff))

# Subgraphic
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.plot(train_x, train_y, 'o')

ax2 = fig.add_subplot(2,2,2)
ax2.plot(train_z, train_y,'o')

x = np.linspace(-3,3,100)
ax3 = fig.add_subplot(2,2,3)
ax3.plot(train_z, train_y, 'o')
ax3.plot(x,f(x))
plt.show()  

print('{0:=^50}'.format('End of source'))
print(__file__)