'''
Tomcat access log analysis
Created on 2019. 6. 5
@author: HRKim
'''
#
#<Valve className="org.apache.catalina.valves.AccessLogValve" directory="logs" 
# pattern="%h %l %u %t &quot;%r&quot; %s %b" 
#
# %h IP Address of the client
# %l is RFC1413 identity 
# %u is the user id, determined by the HTTP authentication
# %t is the time 
# \"%r\" is the request string, formatted as "method resource protocol"
# %s is the status code
#
# prefix="localhost_access_log." suffix=".txt"/>
#192.1.1.5 - - [05/Jun/2019:09:22:45 +0900] "POST /Rel_CurStateSrch.do HTTP/1.1" 200 375

import pandas as pd

web_log_file = '../data/weblog01.txt'

df_web_log = pd.read_csv(web_log_file
                         #, sep=''
                         , sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])'
                         # ?=...  ...에 매치되는 정규식에 매치되어야 하며 조건이 통과되어도 문자열이 소비되지 않음
                         # ?:
                         # ?!...    ...에 매치되는 정규식에 매치되지 않으며  조건이 통과되어도 문자열이 소비되지 않음
                         , engine='python'
                         , usecols = [0,3,4,5,6,7,8]
                         , names = ['ip','Identity','UserID','access_time', 'url', 'status', 'size']
                         , na_values = '-'      # '-' 값을 NaN으로 처리 
                         , header = None )

print(df_web_log)
#dft = df_web_log.loc[(df_web_log['status'] == 404)]
#print(dft['url'],dft['ip'])

#df_grp = df_web_log.groupby(['url']).count()
df_grp = df_web_log.reset_index().groupby(['url'],as_index=False).count()
print(df_grp['url'])
