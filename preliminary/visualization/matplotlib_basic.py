'''
Created on May 24, 2019

@author: hrkim
'''
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(360, step=20)
y = np.sin(x * 2 * np.pi / 360)

# 그래프 크기 정하기 
plt.figure(figsize=(6,3))

plt.plot(x,y)

plt.show()

# 한번에 여러개의 그래프 그리기 
for i in range(1,5):
    plt.subplot(2,2,i)
    plt.text(0.5,0.5,str('Subplot %i' %i), ha='center')
    plt.grid(True)

plt.show()    

###
colors = [ 'red', 'green', 'blue', 'gray']
linestyles = [ '-', '--', ':', '-.']

for i in range(1,5):
    plt.subplot(2,2,i)
    plt.plot(x,y,color=colors[i-1]
             , linewidth = i
             , linestyle = linestyles[i -1]) 
    plt.grid(True)
    
plt.show()

### Marks 
colors = [ 'red', 'green', 'blue', 'gray']
linestyles = [ '-', '--', ':', '-.']
markersstyles = ['^', 'x', 'v', 'o']

for i in range(1,5):
    plt.subplot(2,2,i)
    plt.plot(x,y,color=colors[i-1]
             , linewidth = i
             , linestyle = linestyles[i -1]
             , marker = markersstyles[i-1]) 
    #
    plt.legend(labels = str(i))
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.tight_layout()
    
    # 축단위와 범위 지정하기
    plt.yticks([-1.0,0.0,1.0],['Low', 'Mid', 'High'])
    plt.xticks([0,100,200,300],['0 \nsec', '100\nsec', '200\nsec', '300\nsec'])
    plt.xlim(-50, 350)
    plt.ylim(-1.5, 1.5)
    
    #
    plt.grid(True)
    
    #
    plt.annotate('Arrow test', xy = (200,0)
                 , arrowprops = {'color': 'red'})
    
plt.show()

plt.savefig('plot1.jpg')
