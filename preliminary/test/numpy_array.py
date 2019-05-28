'''
Created on 2019. 5. 22.

@author: HRKim
'''
import numpy as np

a = [ 1,2,3]
b = [ 4,5,6]

print(a+b)      # result : [ 1,2,3,4,5,6] 
# print(a-b)    # unsupported operand list
# print(a*b)    # TypeError: can't multiply sequence by non-int of type 'list'
# print(a/b)    # TypeError: unsupported operand type(s) for /: 'list' and 'list'


x = np.array([ 1,2,3])
y = np.array([ 4,5,6])

print(x+y)      # [5 7 9]
print(x-y)      # [-3 -3 -3]
print(x*y)      # [ 4 10 18]
print(x/y)      # [0.25 0.4  0.5 ]

print('{0:=^50}'.format('Array1'))
row1 = [1.]
row2 = [2.]
lst1 = np.array([row1, row2])
print(lst1.shape, lst1)

print('{0:=^50}'.format('Array2'))
w0 = [1.0]
w1 = [2.0]
lst2 = np.array([[w0],[w1]])
print(lst2.shape, lst2)

print('{0:=^50}'.format('Array3'))
w0 = [1.0]
w1 = [2.0]
lst3 = np.array([[w0[0]],[w1[0]]])
print(lst3.shape, lst3)
