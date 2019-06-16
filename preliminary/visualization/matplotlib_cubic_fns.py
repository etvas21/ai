'''
Created on Jun 6, 2019

@author: hrkim
'''
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # matplotlib의 2D그래프를 3D 그래프로 그려
import sys

fig = plt.figure()
# Axes3D축을 fig에 추가. 
# 111은 그려질 서브플롯의 위치로 3자리 정수이며, 111은 1, 1, 1과 동일
# 또한 모든 정수가 10보다 작아야 합니다.
# projection='3d'는 객체가 그래프에 투영될 방법
# projection = [ None, 'mollweide', 'polar', 'rectilinear' ]
ax = fig.add_subplot(111, projection='3d')

x = np.array([x for x in range(-20,30)])
y = np.array([x for x in range(-20,30)])

z = ((x-1.0)**3)*((y+1.0)**3)

ax.plot(x,y,z)
#ax.plot_surface(x,y,z , cstride=4, alpha=0.4, cmap=cm.jet)

plt.xlabel('X value')
plt.ylabel('y label')

a = input('계속하려면 Enter Key를 누리고, 종료를 하려면 1을 입력하세 : ')

if a == '1':
    sys.exit(1)
    
plt.show()
