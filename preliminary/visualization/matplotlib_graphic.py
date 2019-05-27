'''
Created on May 24, 2019

@author: hrkim
'''
import pandas as pd
import matplotlib.pyplot as plt

data = [('None', 'ItemA','ItemB')
        , (2013,100,50)
        , (2014,150, 100)
        , (2015,75,100)]

df = pd.DataFrame(data)
df1 = pd.DataFrame(data, columns=['None', 'ItemA', 'ItemB'])
df2 = df1.drop(0)
df3 = df2.set_index('None')

print('{:=^50}'.format('df'))
print(df)
print('{:=^50}'.format('df1'))
print(df1)
print('{:=^50}'.format('df2'))
print(df2)
print('{:=^50}'.format('df3'))
print(df3)

s1 = df3['ItemA']
print(s1)
ax1 = s1.plot()

s2 = df2['ItemB']
ax2 = s2.plot()

ax3 = df3.plot()

ax3.set_xticks(df3.index)
plt.show()
