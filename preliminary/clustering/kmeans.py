'''
clustering 알고리즘
    KMeans, DBSCAN, Hierarchical clustering, Spectral Clustering


Created on 2019. 5. 22.
@author: HRKim
[출처:https://bcho.tistory.com/1203]
'''

from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core import algorithms

iris = datasets.load_iris()

labels = pd.DataFrame(iris.target)
labels.columns = ['labels']

data = pd.DataFrame(iris.data)
data.columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
data = pd.concat([data,labels],axis=1)

feature = data[['Sepal length', 'Sepal width']]
feature.head()

# Create model and prediction
model = KMeans(n_clusters=3, algorithm='auto')
model.fit(feature)

predict = pd.DataFrame(model.predict(feature))
predict.columns=['predict']

# concatenate labels to df as a new column
r = pd.concat([feature, predict],axis=1)

# visualization
plt.scatter(r['Sepal length'],r['Sepal width'],c=r['predict'],alpha=0.5)

centers = pd.DataFrame(model.cluster_centers_,columns=['Sepal length','Sepal width'])

center_x = centers['Sepal length']

center_y = centers['Sepal width']

plt.scatter(center_x,center_y,s=50,marker='D',c='r')

plt.show()






print('{0:=^50}'.format('End of source'))  
