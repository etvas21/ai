
'''
Method of Least Square(최소제곱)
    y = ax + b
    a = ( x- x의평균)(y - y의평균)의 합 / ( x - x의평균 )의 합의 제곱
    b = y의평균 - ( x의평균 * a )
    
Created on 2019. 5. 20.
@author: HRKim
'''

import numpy as np

x = [ 2, 4, 6, 8]
y = [ 81, 93, 91, 97]

mx = np.mean(x)
my = np.mean(y)

# 분모
divisor = sum([(i - mx)**2 for i in x])

#
def top(x,mx,y,my):
    d = 0
    for i in range(len(x)):
        d +=(x[i] - mx)*(y[i]-my)
    return d

# 분자
dividend = top(x, mx, y, my)

print('분모 :', divisor)
print('분자 :', dividend)

# 
a = dividend / divisor
b = my - ( mx * a )

print('기울기  a = ', a)
print('y 절편 b = ',b)

print('{0:=<20}'.format('Result'))
print('y = {}x + {} '.format(a,b))            

