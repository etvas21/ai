'''
Created on 2019. 5. 20.

@author: HRKim
'''
import numpy as nu

dataset =  nu.loadtxt('../data/pima-indians-diabetes.csv',delimiter=',')

X = dataset[:,0:8]
Y = dataset[:,8]

print('===== X {:=>20}'.format(''))
print(X)
#
print('===== Y {:=>20}'.format(''))
print(Y)