'''
Created on 2019. 5. 20.

@author: HRKim
'''
'''
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
mode : 파일이 열리는 모드
    'r' : 읽기 용으로 열림 (기본값)
    'w' : 쓰기 위해 열기, 파일을 먼저 자른다.
    'x' : 베타적 생성을 위해 열리고, 이미 존재하는 경우 실패
    'a' : 쓰기를 위해 열려 있고, 파일의 끝에 추가하는 경우 추가합니다.
    'b' : 2진 모드(바이너리 모드)
    't' : 텍스트 모드 (기본값)
    '+' : 업데이트 (읽기 및 쓰기)를 위한 디스크 파일 열기
    'U' : 유니버설 개행 모드 (사용되지 않음)
    buffering : 버퍼링끄기는 0(바이너리모드에서만 동작
'''

# open file
f = open('../data/섬집아이.txt', mode='rt', encoding='utf-8')

# read
print('{:=>20}'.format('read'))

for i in range(3):
    print(i,f.read(10))


# 파일포팅트를 맨앞으로 이동
print('{:=>20}'.format('seek and read'))
f.seek(0)
print(f.read(10))


# 파일을 Line별로 읽기
print('{:=>20}'.format('readline'))
f.seek(0)

for i in range(5):
    print(f.readline())


# 파일을 읽어 list에 저장
print('{:=>20}'.format('readlines'))
f.seek(0)
data = f.readlines()

for v in data:
    print(v)


# 파일을 읽어 list에 저장
print('{:=>20}'.format('Iterable'))
f.seek(0)

for line in f:
    print(line)

f.close()

##### try ~ finally
'''
   예외가 발생을 하여도 반드시 파일을 close를 함.
'''
print('=== try~ finally {:=>20}'.format(''))
try:
    f = open('../data/섬집아이.txt', mode='rt', encoding='utf-8')
    print(f.readline())
finally:
    f.close    

##### with
'''
with로 파일을 open하면 with block이 종료가 되면 파일을 자동적으로 close  
'''
print('=== with {:=>20}'.format(''))
with open('../data/섬집아이.txt', mode='rt', encoding='utf-8') as f:
    print(f.readline())
    