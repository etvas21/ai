'''
Created on May 22, 2019

@author: hrkim
[출처:https://blog.naver.com/ackbary/221365904494]
'''

from sklearn.cluster import KMeans # scikit-learn(사이킷런) 
# import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['x','y'])

df.loc[0] = [2,3]
df.loc[1] = [2,11]
df.loc[2] = [2,18]
df.loc[3] = [4,5]
df.loc[4] = [4,7]
df.loc[5] = [5,3]
df.loc[6] = [5,15]
df.loc[7] = [6,6]
df.loc[8] = [6,8]
df.loc[9] = [6,9]
df.loc[10] = [7,2]
df.loc[11] = [7,4]
df.loc[12] = [7,5]
df.loc[13] = [7,17]
df.loc[14] = [7,18]
df.loc[15] = [8,5]
df.loc[16] = [8,4]
df.loc[17] = [9,10]
df.loc[18] = [9,11]
df.loc[19] = [9,15]
df.loc[20] = [9,19]
df.loc[21] = [10,5]
df.loc[22] = [10,8]
df.loc[23] = [10,18]
df.loc[24] = [12,6]
df.loc[25] = [13,5]
df.loc[26] = [14,11]
df.loc[27] = [15,6]
df.loc[28] = [15,18]
df.loc[29] = [18,12]



### visualize data point
sns.lmplot('x','y'
           , data=df
           , fit_reg=False
           , scatter_kws={'s':200})

plt.title('Example before applying K-Means algorithm')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
###
# convert dataframe to numpy array
points = df.values

# initial centroids point를 지정하지 않으면 kmeans++로 자동지정 
kmeans = KMeans(n_clusters=4).fit(points)
kmeans.cluster_centers_

df['cluster'] = kmeans.labels_

sns.lmplot('x','y', data=df
           , fit_reg=False
           , scatter_kws={'s':100}
           , hue='cluster')

plt.title('Example after applying K-Means algorithm')
plt.xlabel('x')
plt.ylabel('y')

plt.show()

