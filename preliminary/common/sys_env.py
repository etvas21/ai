'''
Created on 2019. 5. 29.

@author: HRKim
'''
# for font
import platform
import matplotlib
from matplotlib import font_manager
from matplotlib import rc

# matplotlib에서 한글을 사용할 수 있도록함.
def set_matplotlib_kor():
    # 음수기호를 표시되도록 하기 위함
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 한글표현 
    if platform.system() =='Windows':
        font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
        rc('font', family=font_name)
    else:
        #font_name = font_manager.FontProperties(fname='/Library/Fonts/AppleGothic.ttf').get_name()
        #rc('font', family=font_name)
        rc('font', family='AppleGothic')
    
