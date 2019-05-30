'''
Created on 2019. 5. 29.

@author: HRKim
'''
import pandas as pd
import numpy as np

data = {
    "도시": ["서울", "서울", "서울", "부산", "부산", "부산", "인천", "인천"],
    "연도": ["2015", "2010", "2005", "2015", "2010", "2005", "2015", "2010"],
    "인구": [9904312, 9631482, 9762546, 3448737, 3393191, 3512547, 2890451, 263203],
    "지역": ["수도권", "수도권", "수도권", "경상권", "경상권", "경상권", "수도권", "수도권"]
}
columns = ["도시", "연도", "인구", "지역"]
df1 = pd.DataFrame(data, columns=columns)
print(df1)

print(df1.pivot("도시", "연도", "인구"),'\n')


np.random.seed(0)
df2 = pd.DataFrame({
    'key1': ['A', 'A', 'B', 'B', 'A'],
    'key2': ['one', 'two', 'one', 'two', 'one'],
    'data1': [1, 2, 3, 4, 5],
    'data2': [10, 20, 30, 40, 50]
})
print(df2)

groups = df2.groupby(df2.key1)
print('\n',groups.groups)
print('\n',groups)

print('====sum\n',df2.groupby(df2.key1)['data1'].sum())


#a= df2.groupby(['key1','key2']).size().reset_index(name='mean')
a= (df2.groupby(['key1','key2'])['data1','data2'].mean()).reset_index()
print('\n\n',a.info(),'\n==key1 : ',a['key1'], '\n===data2 : ',a['data2'])

b= df2.reset_index().groupby(['key1','key2'],as_index=False)['data1','data2'].mean()
print('\n====b====\n',b)
print('\n====b====\n',b['data2'])

'''
analytic function 다수 사용하기
'''
print('\n\n', df2.groupby(['key1','key2']).agg(['mean','count']))
