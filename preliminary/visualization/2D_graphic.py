'''
Created on Jun 6, 2019
@author: hrkim
'''
import common.sys_env as env
import numpy as np
import matplotlib.pyplot as plt

env.set_matplotlib_kor()
plt.rcParams['axes.grid'] = True    # grid를 default로 true

#11111
def f(x):
    return x*(x-2)

# linespace(x1,x2,n): 시작 x1,  끝 x2에서 n개의 숫자를 생
x= np.linspace(-5,5,10)
plt.plot(x,f(x))
plt.show()

#22222
def f2(x):
    return 2*x**3 + x**2 - 5*x

x = np.linspace(-3,3,100)
plt.plot(x,f2(x))
plt.show()

#33333
def f3(x,w):
    return (x-w)*x*(x+w)

x= np.linspace(-3,3,100)

plt.plot(x,f3(x,2), color='blue', label='서울')
plt.plot(x,f3(x,4), color='red', label='부산')
plt.plot(x,f3(x,6), color='green', label='대전')
plt.ylim(-30,30)    # 화면에 표시할 x,y축의 범위를 설

plt.legend(loc='upper left')      # label 출력 및 출력할 위치 지정 
#plt.grid(True)                  # 구분선을 표현, params로 지정하여 별도지정하지 않아도  

plt.title('그래프 연습')             # title 
plt.xlabel('x라벨')
plt.ylabel('y라벨')

plt.show()

#44444
plt.figure(figsize=(10,5))   # 그래프 figure 크기 지정 가로 10,세로 5
plt.subplots_adjust(wspace=1, hspace=0.5)    # 그래프 간격 지정 
line_color =np.array(['red','blue','goldenrod','green','orange', 'purple', 'chocolate'])
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.title(i+1)
    plt.plot(x,f3(x,i), color= line_color[i])
    plt.ylim(-30,30)
    #plt.grid(True)    # default로 지정을 하여 지정하지 않아도 
    
plt.show()


