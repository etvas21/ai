'''
Created on 2019. 5. 20.

@author: HRKim
'''
import pandas as pd

df_pre = pd.read_csv('../data/wine.csv',header=None)

print(df_pre.shape)


# sample 
#    n=5: 5개의 Row를 Random하게 return
#    frac: 0<frac<=1  전체 Row에서 몇 지정한 비율로 Return
#        frac=1 > 모든 데이터 반환
df = df_pre.sample(frac=0.5)
print(df.shape)




