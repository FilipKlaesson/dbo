from src.benchmark_functions_2D import *
from matplotlib.ticker import MaxNLocator

import numpy as np
from matplotlib import cm
from scipy.stats import norm
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from scipy import interpolate

class ScalarFormatterForceFormat(ticker.ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"
fmt = ScalarFormatterForceFormat()
fmt.set_powerlimits((0,0))
fmt.useMathText = False

N = 500

# Manual
xmin = -10
xmax = 10
ymin = -10
ymax = 10
function = lambda x: 10*np.exp( -( (x[0] - 3)**2/(2*3**2) + ( x[1])**2/(2*3**2)) ) + \
                    10*np.exp( -( (x[0] + 5)**2/(2*3**2) + ( x[1] + 7)**2/(2*3**2)) ) + \
                    12*np.exp( -( (x[0] + 5)**2/(2*3**2) + ( x[1] - 1 )**2/(2*3**2)) )

# Benchmark
fun = Rastrigin()
function = lambda x: 10**(0)*-1*fun.function(x)
[[xmin, xmax], [ymin, ymax]] = fun.domain

x = np.linspace(xmin,xmax,N)
y = np.linspace(ymin,ymax,N)

X, Y = np.meshgrid(x, y)

F = np.array([function(input) for input in zip(X,Y)])

Fmax = np.max(F)
Fmin = np.min(F)

plt.figure(figsize=(10,10))
ax = plt.subplot(1, 1, 1)
cp = plt.contourf(X, Y, F, np.linspace(Fmin,Fmax,500), cmap = cm.coolwarm)
for c in cp.collections:
    c.set_edgecolor("face")

ax.scatter(fun.arg_min[:,0], fun.arg_min[:,1], marker='x', c='gold', s=50)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.xticks(np.linspace(xmin,xmax,5), rotation=0, fontsize=15)
plt.yticks(np.linspace(ymin,ymax,5), rotation=0, fontsize=15)

plt.xlabel(r"$x$", fontsize=15)
plt.ylabel(r"$y$", fontsize=15, rotation=0)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#plt.show()
plt.savefig('Rastrigin-2D.png', bbox_inches='tight')

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')

#cp = ax.plot_trisurf(X.flatten(), Y.flatten(), F.flatten(), cmap=cm.coolwarm, antialiased=False, shade = False, alpha = 1)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.xticks(np.linspace(xmin,xmax,5), rotation=0, fontsize=15)
plt.yticks(np.linspace(ymin,ymax,5), rotation=0, fontsize=15)


plt.setp(ax.get_xticklabels()[0], visible=False)
plt.setp(ax.get_yticklabels()[0], visible=False)

ax.set_zticks(np.linspace(-120,120,5))
ax.zaxis.set_tick_params(labelsize=15)
#ax.text2D(0.12, 0.73, '$\\times 10^{6}$', transform=ax.transAxes, fontsize=15)
#ax.w_zaxis.set_major_formatter(fmt)
ax.labelsize = 15

ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.zaxis.labelpad = 20

ax.tick_params(pad = 10)

ax.set_xlabel(r"$x$", fontsize=15)
ax.set_ylabel(r"$y$", fontsize=15)
ax.set_zlabel(r"$f$", fontsize=15)

ax.zaxis.offsetText.set_fontsize(15)


tmp_planes = ax.zaxis._PLANES
ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                    tmp_planes[0], tmp_planes[1],
                    tmp_planes[4], tmp_planes[5])

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

ax.view_init(40, 225)

ax.dist = 11

#plt.show()
#plt.savefig('Bird-3D.png', bbox_inches='tight')
