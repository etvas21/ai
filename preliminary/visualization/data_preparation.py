'''
데이터 전처리( data preparation )
Created on 2019. 5. 22.

@author: HRKim
[출처:모두의딥러링]
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/pima-indians-diabetes.csv',
               names = ["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])

# Graphic의 색상 구성
# 상관도가 높을수롤 숫자가 높으며, 밝게 표현됨
# colormap 참조: https://matplotlib.org/users/colormaps.html
# class-plasma 
colormap = plt.cm.binary
plt.figure(figsize=(12,12))     # Graphic 크기

sns.heatmap(df.corr()
            , linewidths=0.1
            , vmax = 0.5
            , cmap=colormap
            , linecolor='white'
            , annot=True)
plt.show()

# 위에서 확인한 결과를 이용하여
# class와 plasma만을 그래프로 표현
# 당뇨병환자(class=1)의 경우 plasma항목의 수치가 150 이상인 경우가 많음을 확인.
grid = sns.FacetGrid(df, col='class')
grid.map(plt.hist, 'plasma', bins=10)
plt.show()


