'''
Created on May 28, 2019

@author: hrkim
'''
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ts = pd.Series(np.random.randn(1000)
               , index=pd.date_range('1/1/2000', periods=1000))
ts = ts.consume()

df = pd.DataFrame(np.random.randn(1000,4)
                , index=ts.index
                , columns=list('ABCD'))

df = df.consume()
plt.figure()
plt.show()                