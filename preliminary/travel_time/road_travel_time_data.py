'''
Created on 2019. 5. 28.
@author: HRKim
'''
import tensorflow as tf
import pandas as pd
import datetime as dt

'''
# Data Analysis
1. read file
2. Check shape, head
3. Extract only the column need
4. Check head/tail
5. Eye check data type of each columns
6. Check Data range
7. Check shape after eliminate value of 'NaN'
'''

# 1. 변수선언
traffic_col_headers= ["depature_day", "depature_time", "depature_office", "arrival_office", "car_type", "lead_time",'unused']
traffic_col_dtypes = {'depature_day':'str', 'depature_time':'int64'
             , "depature_office":'int64',"arrival_office":'int64'
             , "car_type":'int64', "lead_time":'float64', 'unused': 'str'}
# car type
#    승용자(1종), 버스(2종), 화물차(3~12종)
# 기흥:105, 서울:101, 부산(경)639

# 2. 데이터 가져오기
print('{0:=^50}'.format('...Load Data...'))
df_traffic = pd.read_csv('../../../EX_data_영업소간통행시간_TCS_11_03_02_100100.txt'
                 , sep='|' 
                 , header=None
                 , names= traffic_col_headers
                 , dtype= traffic_col_dtypes
                 )

def preview_data(dfx):
    # 3. 데이터 간략보기
    #    unused column 삭제 대상
    print('{0:=^50}'.format('Preview data'))
    print(dfx.head(5))
    print(dfx.tail(5))

    # 4. 자료구조 파악
    print('{0:=^50}'.format('Review data structure'))
    print(df_traffic.info())

    # 5. 숫자 데이터의 데이터 분포 확인
    print('{0:=^50}'.format('dscribe'))
    print(df_traffic.describe())
    
    # 6. column, index 명칭 확인
    print('{0:=^50}'.format('df_traffic.columns.values'))
    print(df_traffic.columns.values)

    print('{0:=^50}'.format('df_traffic.index.values'))
    print(df_traffic.index.values)

def delete_column(dfx):
    # 7. delete unnecessary item
    print('{0:=^50}'.format('Delete column'))
    dfx.drop(['unused'], axis=1,inplace=True)
    
    print(dfx.columns.values)
    
    return dfx

def change_column_name(dfx):    
    # 8. change column name
    print('{0:=^50}'.format('Change column name'))
    #dataframe.rename(columns={'변경전명칭':'변경후명칭'})
    #print('{0:=^50}'.format('dfx.columns.values'))
    #print(dfx.columns.values)
    return dfx

def add_column(dfx):
    # 9. Add new column
    # 위 함수 change_column_name에서 return을 하지 않고 실행을 하면
    # AttributeError: 'NoneType' object has no attribute 'assign' 
    # error 발생
    # [solution] 위 함수에 'return dfx'를 추가하여 신규 컬럼 처리를 하였음.
    
    print('{0:=^50}'.format('Add column'))
    # column 추가방식 1: dfx['year'] = dfx['depature_day'].astype(str).str[0:4]
    dfx = dfx.assign(week_day = pd.to_datetime(dfx['depature_day']).dt.weekday)
    dfx['driving_time'] = dfx['lead_time'] / 60
        
    print(dfx.info())
    
    return dfx

def change_value(dfx):    
    print('{0:=^50}'.format('Change value'))
    return dfx

def analysis_data(dfx):
    print('{0:=^50}'.format('Analysis data'))
    
    missing_df = df_traffic.isnull()    
    print(missing_df.head(5))
    
    return dfx

def final_delete(dfx):
    print('{0:=^50}'.format('Delete column of new data file'))

    dfx.drop(['depature_office', 'arrival_office', 'car_type', 'lead_time'], axis=1,inplace=True)
    
    print(dfx.columns.values)
    
    return dfx    
###
preview_data(df_traffic)

df_traffic = delete_column(df_traffic)
df_traffic = change_column_name(df_traffic)
df_traffic = add_column(df_traffic)
df_traffic = change_value(df_traffic)
df_traffic = analysis_data(df_traffic)

# filter data
df_traffic = df_traffic.loc[(df_traffic['car_type'] == 1)
                            & (df_traffic['depature_office'] == 101)
                            & (df_traffic['arrival_office'] == 105)]     


# save file
print('{0:=^50}'.format('New data file'))
df_traffic = final_delete(df_traffic)
print(df_traffic.info())

df_traffic.to_csv('../data/EX_data_영업소간통행시간_201904_101_105.txt', sep='|', header=False)

print('{0:=^50}'.format('End of source'))
print(__file__)