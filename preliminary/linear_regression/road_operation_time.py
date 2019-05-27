'''
Created on 2019. 5. 27.

@author: HRKim
'''
import pandas as pd






col_headers= ["depature_day", "depature_time", "depature_office", "arrival_office", "car_type", "lead_time",'unused']
col_dtypes = {'depature_day':'str', 'depature_time':'int64'
             , "depature_office":'int64',"arrival_office":'int64'
             , "car_type":'int64', "lead_time":'float64', 'unused': 'str'}

df_traffic = pd.read_csv('../data/EX_data_영업소간통행시간_TCS_11_03_02_100100.txt'
                 , sep='|' 
                 , header=None
                 , names= col_headers
                 , dtype= col_dtypes
                 )

print(df_traffic.columns)
# car type
#    승용자(1종), 버스(2종), 화물차(3~12종)
# 기흥:105, 서울:101
# depature_office를 str으로 read를 하고 filter를 '101'로 하면 filter가 되지 않아서
# depature_office를 int64로 지정을 하고 처리하여 정상적으로 처리됨
# 아마도, str로 읽으면 앞뒤에 space가 있는것 같음.
#dfx = df_traffic.loc[df_traffic.car_type == '1']
dfx = df_traffic.loc[(df_traffic['car_type'] == 1)
                     & (df_traffic['depature_office'] == 101)
                     & (df_traffic['arrival_office'] == 105)]
  
# Data Analysis
'''
1. read file
2. Check shape, head
3. Extract only the column need
4. Check head/tail
5. Eye check data type of each columns
6. Check Data range
7. Check shape after eliminate value of 'NaN'
'''

def check_head():
    # 2. Check shape, head
    print('{0:=^50}'.format('Analysis shape'))
    print(df_traffic.shape)
    
def check_head_tail():
    # 처음 5줄 보기
    print('{0:=^50}'.format('Analysis Data (head)'))  
    print(df_traffic.head(5))
    print('{0:=^50}'.format('Analysis Data (tail)'))
    print(df_traffic.tail(5))

def check_info():
    # data 정보 확인
    print(df_traffic.info())

    # 각 정보별 특징을 더자세히( count, mean, std, max ..)
    print(df_traffic.describe())

####################################
# learning
####################################


####################################
# applying
####################################


check_head()
check_head_tail()
check_info()

print('{0:=^50}'.format('dfx'))
print(dfx.head(5))
print(dfx.shape)

# HYPORTHESIS
#    w1: 요일
#    w2: 출발시간
#    w3: 기후 (0~1)
#    w4: 공사 (0~1)
#    hypothesis: 소요시간
#hypothesis = w1*x1 + w2*x2 + b


print('{0:=^50}'.format('End of source'))
print(__file__)
 No newline at end of file
