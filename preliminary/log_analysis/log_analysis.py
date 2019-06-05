'''
Tomcat access log analysis
Created on 2019. 6. 5
@author: HRKim
'''
#
#<Valve className="org.apache.catalina.valves.AccessLogValve" directory="logs" 
# pattern="%h %l %u %t &quot;%r&quot; %s %b" 
# prefix="localhost_access_log." suffix=".txt"/>
#

import pandas as pd


web_log_file = '../data/weblog01.txt'

df_web_log = pd.read_csv(web_log_file
                         #, sep=''
                         , engine='python'
                         #, usecols = []
                         #, names = ['']
                         , na_values = '-'
                         , header = None )

print(df_web_log)

