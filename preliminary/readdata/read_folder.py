'''
Created on 2019. 5. 23.

@author: HRKim
[출처:https://3months.tistory.com/203?category=753896]
'''
from os import listdir
from os.path import isfile, join

files = [ f for f in listdir('../data') if isfile(join('../data', f))]

print( files)

filter_files = [ x for x in files if x.find('txt') !=-1]

print(filter_files)
