'''
@author: hrkim
Created on Jun 13, 2019
'''
import pandas as pd
import numpy as np
import sys

'''
1. Create Dataframe
2. DDL
3. DML
4. Additional(?) function
'''
#####################################
# Create  Dataframe
#####################################
print('{0:=^30}\n{1}\n{0:=^30}'.format('','Create DataFrame'))

#CASE 1: list
xdata = [[1,2,3],[4,5,6]]

df = pd.DataFrame(data = xdata)
print('\n{0:[^3} {1} {0:]^3}\n{2}'.format('','Create DataFrame with list',df))

###
xdata = [ [201903, 90,86,93]
         , [201905, None, 89,90]
         , [201907, 92, 90, None] ]

df = pd.DataFrame(data = xdata)
print('\n{0:[^3} {1} {0:]^3}\n{2}'.format('','Create DataFrame with list',df))

###
xdata = [ [201903, 90,86,93]
         , [201905, None, 89,90]
         , [201907, 92, 90, None] ]
xcol =['EVAL_YM', 'KOR', 'ENG', 'MATH']

df = pd.DataFrame(data = xdata, columns = xcol)
xmsg = 'To create a DataFrame by specifying a column name in the list'
print('\n{0:[^3} {1} {0:]^3}\n{2}'.format('' , xmsg , df))

###
xdata = [ [201903, 90,86,93]
         , [201905, None, 89,90]
         , [201907, 92, 90, None] ]
xcol = ['EVAL_YM', 'KOR', 'ENG', 'MATH']
xidx =  ['one','two','three']

df = pd.DataFrame(data = xdata, columns = xcol, index = xidx)
xmsg = 'To create a DataFrame by specifying a column name, index  in the list'
print('\n{0:[^3} {1} {0:]^3}\n{2}'.format('', xmsg ,df))


#CASE 2: nparray
xdata = np.array([[1,2,3],[4,5,6]])

df = pd.DataFrame(data = xdata)
print('\n{0:[^3} {1} {0:]^3}\n{2}'.format('','Create DataFrame with np.array',df))

#CASE 3: dictionary
xdata = {'EVAL_YM' : [201903, 201905,201907]
         , 'KOR' : [90, None, 92]
         , 'ENG' : [86, 89, 90]
         , 'MATH': [93,90,None] }

df = pd.DataFrame(xdata)
print('\n{0:[^3} {1} {0:]^3}\n{2}'.format('','Create DataFrame with dictionary',df))

###
# Create DataFrame with dictionary by specifying and index and column name
# index가 같으면 최종 값으로 지정한다.
xdata = {'KOR':{201903:90, 201905:92}
             , 'ENG': {201903:86, 201903:89, 201903:90}
             , 'MATH': {201903:93, 201905:90, 201907:96}}

df = pd.DataFrame(xdata)

xmsg = 'Create DataFrame with dictionary by specifying and index and column name'
print('\n{0:[^3} {1} {0:]^3}\n{2}'.format('', xmsg, df))

###
# Select index to create DataFrame 
xdata = {'KOR':{201903:90, 201905:92}
             , 'ENG': {201903:86, 201905:89, 201907:90}
             , 'MATH': {201903:93, 201905:90, 201907:96}}

df = pd.DataFrame(xdata, index=[201903,201905])
print('\n{0:[^3} {1} {0:]^3}\n{2}'.format('','Select index to create DataFrame',df))

#CASE 4: Series
df = pd.Series({'UK':'London', 'India':'Delhi', 'USA':'Washington'})
print('\n{0:[^3} {1} {0:]^3}\n{2}'.format('','Create DataFrame with Series',df))

# CASE 5: Create DataFrame using file
xdata_file = '../data/exam_csv.csv'
xcol = ['EVAL_YM', 'KOR', 'ENG', 'MATH']
xidx =  ['one','two','three']

df = pd.read_csv(xdata_file
                 , sep=','
                 , names = xcol           # (옵션) csv 파일의 컬럼명을 지정     
                 , index_col = 0            # (옵션) 1번째 column을 index column으로 지정             
                 , usecols = [0,1,2,3]      # (옵션) Load할 컬럼의 index를 지정 
                 , header = None )
print('\n{0:[^3} {1} {0:]^3}\n{2}'.format('','Create DataFrame using file',df))

#####################################
# Dataframe 구조관리 ( Like DDL)
#####################################
#
# Column 순서 변경 
#
print('#'*30, '\nChange column order\n', '#'*30)
xdata ={'KOR':{201903:90, 201905:92}
             , 'ENG': {201903:86, 201905:89, 201907:90}
             , 'MATH': {201903:93, 201905:90, 201907:96}}

df = pd.DataFrame(xdata)
print('\n=== Original Dataframe to change order of column ===\n', df)


dfx = pd.DataFrame(df, columns =['MATH','KOR','ENG'])
print('\n=== Column 순서 변경 ===\n', dfx)

#
# column 추가
#
print('#'*30, '\nInsert column\n', '#'*30)

df = pd.DataFrame(data=np.array([['201903',90,86,93],['201905',92,89,90],['201907',None,90,96]])
               , columns=['EVAL_YM','KOR','ENG','MATH'])
print('\n=== Original Dataframe to insert column ===\n',df)

df.loc[:,'HIST'] = pd.Series([5,6,6], index=df.index)
print('\n=== Case 1. After insert column ===\n',df)

df['PHY'] = pd.Series([5,6,6], index=df.index)
print('\n=== Case 2. After insert column ===\n',df)

df['CHEM'] = pd.Series(range(len(df)), index=df.index)
print('\n=== Case 3. After insert column with range===\n',df)

df['KOR_GOOD'] = df.KOR > 90
print('\n=== Case 4. df.KOR_GOOD ===\n',df)

#df['ENG_GOOD'] = df.apply(lambda x: x['ENG'] if x['ENG'] > 80 else 0)
df['ENG_GOOD'] = df['ENG'].apply(lambda x: 'A' if x > 90 else 'B')
print('\n=== Case 5. df.ENG_GOOD ===\n',df)
sys.exit(0)

#
# column 삭제
#
print('#'*30, '\nDelete column\n', '#'*30)

xdata ={'KOR':{201903:90, 201905:92}
             , 'ENG': {201903:86, 201905:89, 201907:90}
             , 'MATH': {201903:93, 201905:90, 201907:96}}
df = pd.DataFrame(xdata)
print('\n=== Original Datrframe to del or drop column ===\n',df)

# use drop
df = df.drop('ENG', axis=1)    # axis=0: row, axis=1: column
print('\n=== Case 1. After drop column using drop===\n',df)

# use del
del df['MATH']
print('\n=== Case 2. After delete column using del ===\n',df)

#
# index 컬럼 변경하기
#
print('#'*30, '\nChange index column\n', '#'*30)

df = pd.DataFrame(data=np.array([['201903',90,86,93],['201905',92,89,90],['201907',0,90,96]])
               , columns=['EVAL_YM','KOR','ENG','MATH'])

print('\n=== Original Datrframe ===\n',df)

dfx = df.set_index('EVAL_YM')
print('\n=== Changed datafrom ===\n',dfx)

#TODO
# index column 지정, index column 명 지정
'''
df_test = DataFrame(exam_rslt)
df_test.index.name ='year'
df_test.columns.name ='YEAR'
print('\n=== exam_rslt ===\n',df_test)
print('\n=== exam_rslt ===\n',df_test.values)
'''

#
# 행렬 바꾸기 
#
print('#'*30, '\nTranspose row and column\n', '#'*30)

xdata ={'KOR':{201903:90, 201905:92}
             , 'ENG': {201903:86, 201905:89, 201907:90}
             , 'MATH': {201903:93, 201905:90, 201907:96}}

df = pd.DataFrame(xdata)
print('\n=== Original Datrframe ===\n',df)

dft = pd.DataFrame(df.T)
print('\n=== Transposed Dataframe ===\n',df.T)


#####################################
# data  조직( Like DML )
#####################################
# Read Data
print('#'*30, '\nRead Data\n', '#'*30)

xdata = np.array([[1,2,3],[4,5,6],[7,8,9]])
xcolumns = ['A','B','C']

df = pd.DataFrame(data=xdata, columns = xcolumns )
print('\n=== Original Datrframe ===\n',df)
# oracle의 rowid와 같으므로, 데이터를 내가 지정한 값으로 
# 직접 접근을 하기위하여 index 변경을 함
# 예, 201907 행을 데이터를 일기위해
# default index는 알수 가 없으나, 내가 지정한 index를 사용하면
# 읽기가 용이
# ix: index값이 지정한 값인 데이터 읽기 - index가 integer로만 
#    구성이 되면 index로 접근하지만, 그렇지 않으면 순서대로 접근(iloc와 동)
# loc
# iloc: index와 상관없이 지정한 값의 순서를 보고 데이터 읽기 

print('\n=== Read row data with df[''A''] ===\n', df['A'])
print('\n=== Read row data with df.ix[0,''A''] ===\n', df.ix[0, 'A'])
print('\n=== Read row data with df.loc[:,''A''] ===\n', df.loc[:,'A'])
print('\n=== Read row data with df.loc[0,''A''] ===\n', df.loc[0,'A'])
print('\n=== Read row data with df.iloc[1] ===\n', df.iloc[1])


xdata ={'KOR':{'201903':90, '201905':92}
             , 'ENG': {'201903':86, '201905':89, '201907':90}
             , 'MATH': {'201903':93, '201905':90, '201907':96}}

df = pd.DataFrame(xdata)

print('\n=== Original Datrframe ===\n',df)
print('\n=== df.MATH ===\n',df.MATH)

### todo reset index
#print('\n=== Read row data with ix ===\n', df.ix[201907])
print('\n=== Read row data with loc ===\n', df.loc['201905'])
print('\n=== Read row data with iloc ===\n', df.iloc[1])


print('\n\n', df[(df['KOR'] >= 92.0) & (df['MATH'] <= 93.0)])  


xdata = [['한국기업',20190101, 1000,'CLOSE'],['한국기업',20190101,2000,'REJ'],['한국기업',20190201,3000,'REC']
         ,['우리상사', 20190201, 4000,'mfg'],['우리상사',20190201,4500,'DLV'],['우리상사',20190501,3400,'CLOSE']
         ,['우리상사',20190501,3200,'REJ'],['우리상사', 20190201, 4000,'REC']]                               
xcols = ['CUST', 'ODR_DT','ODR_AMT','ODR_STATUS']                                                  

df = pd.DataFrame( data = xdata, columns = xcols)
print('\n=== Original Datrframe ===\n',df)

print('\n\n', df[(df['ODR_STATUS'] =='CLOSE') & (df['ODR_AMT'] > 2000)])
        
sys.exit(1)

################### TODO ##########################
#
# Update Data
#
print('#'*30, '\nUpdate Data\n', '#'*30)

df.MATH = 90
print('\n=== df.MATH ===\n',df.MATH)

df.MATH = np.arange(len(df))
print('\n=== df.MATH ===\n',df.MATH)

xval = Series([90,70], index=[201903,201907])
df.MATH = xval
print('\n=== df.MATH ===\n',df)

# TODO
#df[(df['ODR_STATUS'] =='CLOSE') & (df['ODR_AMT'] > 2000)]  

# 
df = pd.DataFrame(data=np.array([['201903',90,86,93],['201905',92,89,None],['201907',None,90,96]])
               , columns=['EVAL_YM','KOR','ENG','MATH'])
print('\n=== Original Data ===\n',df)

# 임의 컬럼의  Null 값을 지정된 값으로 변경하기 
print('\n=== To update a columns in a column containing null ===\n',df.fillna(0))

# replace( old_value, new_value)  
print('\n=== To change null to specified value ===\n',df.replace(np.nan, -1))
#
# ROW 추가( index, loc 이용 )
#
print('#'*30, '\nInsert Row using default index\n', '#'*30)

xdata = np.array([[1,2,3],[4,5,6],[7,8,9]])
xcolumns = ['A','B','C']
df = pd.DataFrame(data=xdata, columns = xcolumns )
print('\n=== df. ===\n',df)

df.ix[5] = [10,11,12]
print('\n=== df. ===\n',df)

df.ix[1] = [100,111,122]
print('\n=== df. ===\n',df)


print('#'*30, '\nInsert Row using user index\n', '#'*30)
xdata ={'KOR':{201903:90, 201905:92}
             , 'ENG': {201903:86, 201905:89, 201907:90}
             , 'MATH': {201903:93, 201905:90, 201907:96}}

df = pd.DataFrame(xdata)
print('\n=== df. ===\n',df)
#- if index labeled '2'  exist ,it change with new values.
#- if not, It make an index labeled '2' and add the new values
df.ix[201908] = [60,50,40]
print('\n=== df. ===\n',df)

# This will make an index labeled '2' and add the new values
df.ix[201903] = [60,50,40]
print('\n=== df. ===\n',df)

#
# Append row
#
# index를 고려하지 않고 추가하는 경우
print('#'*30, '\nTo insert Row without considering the index\n', '#'*30)
xdata = np.array([[1,2,3],[4,5,6],[7,8,9]])    # or xdata = [[1,2,3],[4,5,6],[7,8,9]]
xcolumns = ['A','B','C']
df = pd.DataFrame(data=xdata, columns = xcolumns )
print('\n=== Original Dataframe. ===\n',df)

xdata_new = pd.DataFrame(data=[[90,100]], columns=['A','C'])

df = df.append(xdata_new)
print('\n=== Appended Row ===\n',df)

df = df.reset_index(drop=True)
print('\n=== Reset index ===\n',df)

#
# 데이터 수정하기
#
print('#'*30, '\nUpdate data\n','#'*30)
xdata = np.array([['201903',90,86,93],['201905',92,89,90],['201907',0,90,96]])
xcolumns =  ['EVAL_YM','KOR','ENG','MATH']
df = pd.DataFrame(data= xdata, columns= xcolumns)

print('\n=== Original Data ===\n',df)

df.loc[0,'KOR'] = 100
print('\n=== Updated Data ===\n',df)

df.loc[0:1,'ENG'] = 100
print('\n=== Updated Data ===\n',df)

####
#.ix is deprecated. Please use
#.loc for label based indexing or
#.iloc for positional indexing
# loc[n]  :  n값의 행
# loc[:m] : m 값의 행까지
# loc[n:m] : n값의 행부터m값의 행까지  
####
df = pd.DataFrame(np.arange(25).reshape(5,5), 
                      index=list('87cba'),
                      columns=['x','y','z', 'a', 'b'])
print('\n=== Original Data ===\n',df)

df.loc['b','a'] = 99
print('\n=== Updated Data ===\n',df)

df.loc['7':'c','z'] = 100                        
print('\n=== Updated Data ===\n',df)

#TODO
#df['ORD_STATUS'].apply(lambda x: x.lower() if x =='REJ'  else x )  

#TODO
#df['ORD_AMT'] = df['ORD_AMT'].apply(lambda x: x - 2000 if x >3000 else x) 

#
# 데이터 삭제 
#
print('#'*30, '\nDelete data\n','#'*30)
df = pd.DataFrame(np.arange(25).reshape(5,5), 
                      columns=['A','B','C', 'D', 'E'])
print('\n=== Original Data ===\n',df)

#df.index[1]    index 가 1인 값을 가져오기 
df = df.drop(index=1)   # or df = df.drop(1)
print('\n=== Delete Row ===\n',df)


###
df = pd.DataFrame(np.arange(25).reshape(5,5), 
                      index=list('87cba'),
                      columns=['x','y','z', 'a', 'b'])
print('\n=== Original Data ===\n',df)

df = df.drop(index='c')   # or df = df.drop('c')
print('\n=== Delete Row ===\n',df)

#
df = pd.DataFrame(data=np.array([['201903',90,86,93],['201905',92,89,90],['201907',None,90,96]])
               , columns=['EVAL_YM','KOR','ENG','MATH'])
print('\n=== Original Data ===\n',df)

# 임의의 컬럼에 Null이 포함된 행을 삭제하기 
print('\n=== To delete a row in a column containing null ===\n',df.dropna())


#
# 중복 Row 삭제
#
print('#'*30, '\nDelete duplicated  data\n','#'*30)

xdata = [['한국기업',20190101, 1000],['한국기업',20190101,2000],['한국기업',20190201,3000]
         ,['우리상사', 20190201, 4000],['우리상사',20190201,4500],['우리상사',20190501,3400]
         ,['우리상사',20190501,3200],['우리상사', 20190201, 4000]]                               
xcols = ['CUST', 'ODR_DT','ODR_AMT']                                                  

df = pd.DataFrame( data = xdata, columns = xcols)
print('\n=== Original Data ===\n',df)

print('\n=== Boolean status duplicated data ===\n',df.duplicated(['CUST']))

# CUST,ODR_DT 기준으로 중복된 행들에 대하여 True를 표시한다. 
print('\n=== Boolean status duplicated data ===\n',df.duplicated(['CUST','ODR_DT']))

print('\n=== To delete duplicate CUST ===\n',df.drop_duplicates(['CUST']))

# CUST,ODR_DT 기준으로 중복된 행들을  삭제하면서 , 첫번째 중복된 행을  남겨둔다. 
print('\n=== To delete duplicate CUST, ODR_DT ===\n',df.drop_duplicates(['CUST','ODR_DT']))

# CUST,ODR_DT 기준으로 중복된 행을 제거하면서 keep=last에 의해서 중복된 행중 
# 마지막 행을 남겨준다 ( default는 first, 즉 첫번째 중복된 행을 남겨 두는것 )
print('\n=== To leave the final row in a duplicate row and delete a duplicate CUST,ODR_DT ===\n'
      ,df.drop_duplicates(['CUST','ODR_DT'],keep = 'last'))

#
# Mapping data
# 
print('#'*30, 'nTo map data using code master\n','#'*30)

xdata = [['한국기업',20190101, 1000,'CLOSE'],['한국기업',20190101,2000,'REJ'],['한국기업',20190201,3000,'REC']
         ,['우리상사', 20190201, 4000,'mfg'],['우리상사',20190201,4500,'DLV'],['우리상사',20190501,3400,'CLOSE']
         ,['우리상사',20190501,3200,'REJ'],['우리상사', 20190201, 4000,'REC']]                               
xcols = ['CUST', 'ODR_DT','ODR_AMT','ODR_STATUS']                                                  

df = pd.DataFrame( data = xdata, columns = xcols)
print('\n=== Original Data ===\n',df)

dic_code_master = {'ISS':'요청', 'REC':'주문접수', 'REJ':'반품', 'MFG':'생산중', 'DLV':'배송중', 'CLOSE':'주문종료'}


df['ODR_STATUS_NM'] = df['ODR_STATUS'].apply(lambda x: dic_code_master[x.upper()]) 
print('\n=== Mapped  Data ===\n',df)
     
#
# 추가기능 Categories 
# 
print('#'*30, 'nCategoris \n','#'*30)

xdata = [['한국기업',20190101, 1000,'CLOSE'],['한국기업',20190101,2000,'REJ'],['한국기업',20190201,3000,'REC']
         ,['우리상사', 20190201, 4000,'mfg'],['우리상사',20190201,4500,'DLV'],['우리상사',20190501,3400,'CLOSE']
         ,['우리상사',20190501,3200,'REJ'],['우리상사', 20190201, 4000,'REC']]                               
xcols = ['CUST', 'ODR_DT','ODR_AMT','ODR_STATUS']                                                  

df = pd.DataFrame( data = xdata, columns = xcols)
print('\n=== Original Data ===\n',df)

# Column ODR_STATUS를 category type으로 지정을 하고, CD column으로 생성을 한다. 
# ODR_STATUS의 갑중에서 소문자가 있어, 이를 대문자로 변경을 하고 category로 타입을 변경한다.
df['CD'] = df['ODR_STATUS'].apply(lambda x: x.upper()).astype('category') 
print('\n=== Added column of category type ===\n',df, '\n', df['CD'])

# category 값 확인 
print('\n===  Review  category value ===\n',df, '\n', df['CD'].cat.categories)


print('{0:=^50}'.format('End of source'))
print(__file__)
