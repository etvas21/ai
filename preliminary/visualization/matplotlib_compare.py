'''
Created on May 25, 2019

@author: hrkim
'''
import numpy as np
import matplotlib.pyplot as plt


# 1. Assign x, y values
plt.plot([1,2,3],[1,2,3])
plt.show()

# 2개의 그래프 합치기
x = np.arange(1,10,0.1)
y1 = 2 * x + 1
y2 = np.sin(x)


plt.plot(x, y1, label='Y1')
plt.plot(x, y2, linestyle='--', label='sin function')

plt.title('To represent two curve as one graphic')
plt.xlabel('x asix')
plt.ylabel('y axis')
plt.legend()
plt.show()



# prepare data
data = [ (0,3)
        , (1,4)
        , (2,7)
        , (3,9)
        , (4,11)]
#
#hypothesis = 3 * x + b

# Graphic 설
plt.title('Title')
plt.xlabel('X axis')
plt.ylabel('Y axis')
    
plt.plot(data)
plt.show()

####
x1 = np.arange(0,12,1)  # 0부터 5까지 0.1 간격으로 생성 
x2 = np.arange(2,14,1)
y = 5 * x1**2 + 4 * x2 + 3

plt.plot(x1,x2,y)
plt.show()
