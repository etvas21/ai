'''
Created on 2019. 5. 29.

@author: HRKim
'''
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt

import numpy as np

def f(x, y):
    #return 2 * x**2 + 6 * x * y + 7 * y**2 - 26 * x - 54 * y + 107
    return 1.1002*x + 0.3198*y

xx = np.linspace(-3, 7, 100)
yy = np.linspace(-3, 7, 100)
X, Y = np.meshgrid(xx, yy)
Z = f(X, Y)

fig = plt.figure()

ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, linewidth=0.1)
ax.view_init(40, -110)
plt.xlabel('x')
plt.ylabel('y')
plt.title("서피스 플롯(Surface Plot)")
plt.show()