'''
Created on 2019. 5. 22.

@author: HRKim
[출처:모두의딥러링]
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# seed 값 설정
seed =0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('../data/iris.csv', names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

sns.pairplot(df, hue='species')
plt.show()

