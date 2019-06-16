'''
Created on Jun 6, 2019

@author: hrkim
'''
import common.sys_env as env
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

env.set_matplotlib_kor()
plt.rcParams['axes.grid'] = True # Default로 지정 

xn = 50
x = np.linspace(-2,2,xn)
y = np.linspace(-2,2,xn)
z = np.zeros((len(x),len(y)))

def f(x,y):
    r = 2*x**2 + y**2
    
    return r * np.exp(-r)

for i in range(xn):
    for j in range(xn):
        z[j,i] = f(x[i], y[j]) 
        
plt.subplot(2,3,1)        
plt.gray()
plt.pcolor(z)
plt.colorbar()
#plt.show()

#22222
ax=plt.subplot(2,3,2, projection='3d')
xx,yy = np.meshgrid(x,y)

#ax = plt.subplot(1,1,1, projection='3d')
#ax.plot_surface(xx,yy,z)
ax.plot_surface(xx,yy,z
                ,rstride = 1    #
                ,cstride = 1    # 
                ,alpha = 0.3    # 투명도 지정 0~1, 0에 가까울수록 투
                ,color = 'blue',edgecolor = 'black')

#plt.show()

###33333 보는각도(시점) 변환 
ax = plt.subplot(2,3,3, projection ='3d')
#ax = plt.subplot(1,1,1, projection='3d')
#ax.plot_surface(xx,yy,z)
ax.plot_surface(xx,yy,z
                ,rstride = 1    #
                ,cstride = 1    # 
                ,alpha = 0.3    # 투명도 지정 0~1, 0에 가까울수록 투
                ,color = 'blue',edgecolor = 'black')
# 75: 상하회전각도 0-옆에서 본각도, 90-위에서 본각
# -95: 좌우회전각도 양수-시계방향으로 회전, 음수-시계반대방향으로 회
ax.view_init(75,-95)     
#plt.show()

###4444 z 축의 값을 제한 
ax = plt.subplot(2,3,4, projection='3d')
#ax = plt.subplot(1,1,1, projection='3d')

#ax.plot_surface(xx,yy,z)
ax.plot_surface(xx,yy,z
                ,rstride = 5    #
                ,cstride = 5    # 
                ,alpha = 0.3    # 투명도 지정 0~1, 0에 가까울수록 투
                ,color = 'blue',edgecolor = 'black')
# 75: 상하회전각도 0-옆에서 본각도, 90-위에서 본각
# -95: 좌우회전각도 양수-시계방향으로 회전, 음수-시계반대방향으로 회
ax.set_zticks((0,0,2))
ax.view_init(75,-95)     
#plt.show()

###55555 등고선을 표시
plt.subplot(2,3,6)
cont = plt.contour(xx,yy,z,5,colors='black')
cont.clabel(fmt='%3.2f', fontsize=8)

plt.show()


