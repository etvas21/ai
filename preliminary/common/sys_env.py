'''
Created on 2019. 5. 29.

@author: HRKim
'''
# for font
import platform
from matplotlib import font_manager
from matplotlib import rc

# matplotlib에서 한글을 사용할 수 있도록함.
def set_matplotlib_font():
    if platform.system() =='Windows':
        font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
        rc('font', family=font_name)
    else:
        rc('font', family='AppleGothic')
    
