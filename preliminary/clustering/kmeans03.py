'''
Created on 2019. 5. 23.

@author: HRKim
[출처:텐서플로우첫걸음]
'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf


num_points = 2000
vectors_set =[]

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0,0.9)
                            , np.random.normal(0.0,0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5)
                            , np.random.normal(1.0,0.5)])

df = pd.DataFrame({ 'X': [v[0] for v in vectors_set]
                   , 'Y': [v[1] for v in vectors_set]})
        
sns.lmplot( 'X', 'Y'
            , data = df
            , fit_reg = False)

print('{0:=^50}'.format('Befor applying K-Means'))
plt.show()

# 무작위로 생성한 데이터를 상수 텐서로 변환
vectors = tf.constant(vectors_set)
k=4

centroids = tf.Variable(tf.slice( tf.random_shuffle(vectors),[0,0],[k,-1]))

expanded_vectors = tf.expand_dims(vectors,0)
expanded_centroids = tf.expand_dims(centroids,1)

assignments = tf.argmin( 
                tf.reduce_sum( 
                    tf.square( 
                        tf.subtract( expanded_vectors, expanded_centroids)),2),0)

means = tf.concat( [tf.reduce_mean( 
                        tf.gather( 
                            vectors, tf.reshape( 
                                tf.where( 
                                    tf.equal(assignments,c)),[1,-1]))
                                        , reduction_indices=[1]) for c in range(k)],0)

update_centroids = tf.assign(centroids, means)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    
    for step in range(100):
        _, centroid_values, assignment_values = sess.run( [ update_centroids, centroids, assignments])
    data = {'x': [], 'y': [], 'cluster': []}
    
    for i in range(len(assignment_values)):
        data['x'].append(vectors_set[i][0])
        data['y'].append(vectors_set[i][1])
        data['cluster'].append(assignment_values[i])
    df = pd.DataFrame(data)
    
    sns.lmplot('x', 'y'
               , data = df
               , fit_reg = False
               , size=6
               , hue ='cluster'
               , legend=False)
    plt.show()
    print(centroid_values)   
        


print('{0:=^50}'.format('End of source'))  
print(__file__)
