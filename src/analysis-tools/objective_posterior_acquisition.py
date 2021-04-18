from matplotlib.ticker import MaxNLocator

import numpy as np
from matplotlib import cm
from scipy.stats import norm
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from scipy import interpolate


from sklearn.gaussian_process import kernels, GaussianProcessRegressor

class ScalarFormatterForceFormat(ticker.ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"
fmt = ScalarFormatterForceFormat()
fmt.set_powerlimits((0,0))
fmt.useMathText = False

N = 500

# Objective
xmin = -2
xmax = 2
ymin = -2
ymax = 2
function = lambda x: np.sin(3*x[0]) + np.sin(3*x[1])

#Grid
x = np.linspace(xmin,xmax,N)
y = np.linspace(ymin,ymax,N)
X, Y = np.meshgrid(x, y)

F = np.array([function(input) for input in zip(X,Y)])

Fmax = np.max(F)
Fmin = np.min(F)

##################################

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')

cp = ax.plot_trisurf(X.flatten(), Y.flatten(), F.flatten(), cmap=cm.coolwarm, antialiased=False, shade = False, alpha = 0.9)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.xticks(np.linspace(xmin,xmax,5), rotation=0, fontsize=15)
plt.yticks(np.linspace(ymin,ymax,5), rotation=0, fontsize=15)


plt.setp(ax.get_xticklabels()[0], visible=False)
plt.setp(ax.get_yticklabels()[0], visible=False)

ax.set_zticks(np.linspace(-2,2,5))
ax.zaxis.set_tick_params(labelsize=15)
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
plt.savefig('Objective.png', bbox_inches='tight')

##########################################################


n_samples = 10
Xobs = []
Yobs = []
np.random.seed(3)
for params in np.random.uniform(np.array([-2,-2]), np.array([2,2]), (n_samples, 2)):
    Xobs.append(params)
    Yobs.append(function(params))

model = GaussianProcessRegressor(kernel=kernels.RBF(length_scale_bounds=(10**(-1), 10000.0)),
                                 alpha=10**(-10),
                                 n_restarts_optimizer=10
                                )
model.fit(Xobs, Yobs)

x = X.reshape([1,-1])[0]
y = Y.reshape([1,-1])[0]
xxx = np.array([[x[i],y[i]] for i in range(len(x))])
yyy = model.predict(xxx)
Z = np.array(yyy).reshape(X.shape)

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')

cp = ax.plot_trisurf(X.flatten(), Y.flatten(), Z.flatten(), cmap=cm.coolwarm, antialiased=False, shade = False, alpha = 0.9)

XX = []
YY = []
ZZ = []
for x in np.array(model.X_train_):
    XX.append(x[0])
    YY.append(x[1])
for z in model.y_train_:
    ZZ.append(z+0.05)
XX = np.array(XX)
YY = np.array(YY)
ZZ = np.array(ZZ)
ax.scatter(XX,YY,ZZ,color="k",s=10)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.xticks(np.linspace(xmin,xmax,5), rotation=0, fontsize=15)
plt.yticks(np.linspace(ymin,ymax,5), rotation=0, fontsize=15)


plt.setp(ax.get_xticklabels()[0], visible=False)
plt.setp(ax.get_yticklabels()[0], visible=False)

ax.set_zticks(np.linspace(-2,2,5))
ax.zaxis.set_tick_params(labelsize=15)
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
plt.savefig('Surrogate.png', bbox_inches='tight')

#################################

Y_max = np.max(model.y_train_)

mu, sigma = model.predict(xxx, return_std=True)
mu = np.squeeze(mu)

with np.errstate(divide='ignore'):
    Z = (mu - Y_max) / sigma
    expected_improvement = (mu - Y_max) * norm.cdf(Z) + sigma * norm.pdf(Z)
    expected_improvement[sigma == 0.0] = 0
    expected_improvement[expected_improvement < 10**(-100)] = 0

F = expected_improvement

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')

cp = ax.plot_trisurf(X.flatten(), Y.flatten(), F.flatten(), cmap=cm.coolwarm, antialiased=False, shade = False, alpha = 0.95)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.xticks(np.linspace(xmin,xmax,5), rotation=0, fontsize=15)
plt.yticks(np.linspace(ymin,ymax,5), rotation=0, fontsize=15)


plt.setp(ax.get_xticklabels()[0], visible=False)
plt.setp(ax.get_yticklabels()[0], visible=False)

ax.set_zticks(np.linspace(0,0.2,5))
ax.zaxis.set_tick_params(labelsize=15)
#ax.w_zaxis.set_major_formatter(fmt)
ax.labelsize = 15

ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.zaxis.labelpad = 20

ax.tick_params(pad = 10)

ax.set_xlabel(r"$x$", fontsize=15)
ax.set_ylabel(r"$y$", fontsize=15)
ax.set_zlabel(r"$Q$", fontsize=15)

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
plt.savefig('Acquisition.png', bbox_inches='tight')
