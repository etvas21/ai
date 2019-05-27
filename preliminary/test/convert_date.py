'''
Created on 2019. 5. 27.

@author: HRKim
'''

import numpy as np
import datetime as dt

print(dt.date.today())
print('datetime.date(2019,05,27) = ',dt.date(2019,5,27))

# weekday: 0:월,,,,7:일
print('datetime.date(2019,05,27).weekday() = ',(dt.date(2019,5,27)).weekday())


x=[[2019,5,27]
   , [2019,5,28]
   , [2019,5,29]]


for i in range(len(x)):
    x[i].append((dt.date(x[i][0],x[i][1],x[i][2])).weekday())
    print(i, type(i))
    print('datetime.date(2019,05,27).weekday() = ',(dt.date(x[i][0],x[i][1],x[i][2])).weekday())
    
print(x)
#print('datetime.date(2019,05,27).weekday() = ',dt.date(x))


