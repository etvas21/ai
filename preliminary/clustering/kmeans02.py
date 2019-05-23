'''
Created on May 22, 2019

@author: hrkim
[출처:https://blog.naver.com/ackbary/221365904494]
'''
# The sklearn.cluster module gathers popular unsupervised clustering algorithms.
from sklearn.cluster import KMeans # scikit-learn(사이킷런) 
# import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# DataFrame([data, index, columns, dtype, copy])
# index   The index (row labels) of the DataFrame.
# columns The column labels of the DataFrame.
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
# lmplot: Regression plot
# lmplot(x, y, data[, hue, col, row, palette, …])    
#    Plot data and regression model fits across a FacetGrid.
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
# KMeans(n_clusters=8
#        , init=’k-means++’
#        , n_init=10, max_iter=300, tol=0.0001
#        , precompute_distances=’auto’
#        , verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm=’auto’)   
#    n_clusters : int, optional, default: 8
#    The number of clusters to form as well as the number of centroids to generate.
# fit
#    fit(self, X[, y, sample_weight])    Compute k-means clustering.
kmeans = KMeans(n_clusters=4).fit(points)

# cluster_centers_ : array, [n_clusters, n_features]
#        Coordinates of cluster centers. 
#        If the algorithm stops before fully converging (see tol and max_iter), 
#        these will not be consistent with labels_.
'''
[[11.14285714  7.28571429]
 [ 5.53846154  5.53846154]
 [ 7.         17.14285714]
 [15.66666667 13.66666667]]
''' 
print('{0:=^50}'.format('cluster_centers_'))
print(kmeans.cluster_centers_)

# labels_ : Labels of each point
'''
[1 1 2 1 1 1 2 1 1 1 1 1 1 2 2 1 1 0 0 2 2 0 0 2 0 0 3 0 3 3]
'''
print('{0:=^50}'.format('labels_'))
print(kmeans.labels_)

df['cluster'] = kmeans.labels_

sns.lmplot('x','y', data=df
           , fit_reg=False
           , scatter_kws={'s':100}
           , hue='cluster')

plt.title('Example after applying K-Means algorithm')
plt.xlabel('x')
plt.ylabel('y')

plt.show()

print('{0:=^50}'.format('End of source'))  
print(__file__)
