'''
Created on 2019. 5. 20.

@author: HRKim
'''

# Reference: http://pandas.pydata.org

import pandas as pd

df = pd.read_csv('../data/pima-indians-diabetes.csv', 
                 names=["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])

print(df.shape)

# 처음 5줄 보기
print(df.head(5))


# data 정보 확인
print(df.info())

# 각 정보별 특징을 더자세히( count, mean, std, max ..)
print(df.describe())

# 데이터중 지정한 정보와 클래스 만을 출력
print(df[['plasma','class']])
### End of file ###
