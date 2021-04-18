import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

mu = [0, 0]
sigma = [[1, 0.8], [0.8, 1]]
N1 = 1000
N2 = 10
min, max = -2, 2

x = np.linspace(min,max,N1)
y = np.linspace(min,max,N1)
X,Y = np.meshgrid(x,y)

pos = np.array([X.flatten(),Y.flatten()]).T

g = multivariate_normal(mu, sigma)

Z = g.pdf(pos).reshape(N1,N1)
Z = np.array(Z)
Zmin = np.min(Z)
Zmax = np.max(Z)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.contourf(X,Y,Z,np.linspace(Zmin,Zmax,N2),cmap = cm.coolwarm)

ax.plot([0.2, 0.2],[-2, 0.7],'k',linewidth=2)
ax.plot([-2, 0.2],[0.7, 0.7],'k',linewidth=2)
ax.plot([0.2],[0.7],c = 'k',marker = 'o',markersize = 8)

ax.set_xlabel(r"$f(x_1)$", rotation=0, fontsize=30, labelpad=10)
ax.set_ylabel(r"$f(x_2)$", rotation=0, fontsize=30, labelpad=40)

plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    left = False,
    labelbottom=False,
    labelleft=False)

#plt.show()
plt.savefig('bivariate-gaussian-2.pdf', bbox_inches='tight')

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

ax.plot([0.23, 0.23],[-2, 0.2],'k',linewidth=2)
ax.plot([0.66, 0.66],[-2, 0.7],'k',linewidth=2)
ax.plot([0, 0.95],[0, 0],'k--',linewidth = 0.5)
ax.plot([0.23],[0.2],c = 'k',marker = 'o',markersize = 8)
ax.plot([0.66],[0.7],c = 'k',marker = 'o',markersize = 8)

plt.xticks([0.23, 0.66], [r"$x_1$", r"$x_2$"], rotation=0, fontsize=30)
plt.yticks([2], [r"$f(x)$"], rotation=0, fontsize=30)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.tick_params(axis=u'both', which=u'both',length=0)

plt.annotate(r"$\mu$",xy=(0.96, -0.05),fontsize=20)

plt.xlim(0, 1)
plt.ylim(-2, 2)

#plt.show()
plt.savefig('bivariate-gaussian-1.pdf', bbox_inches='tight')
