import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ExpSineSquared)
from matplotlib.lines import Line2D


kernel = RBF(length_scale=1.0, length_scale_bounds='fixed')
gp = GaussianProcessRegressor(kernel=kernel)

# Generate data and fit GP
rng = np.random.RandomState(1337)
X = rng.uniform(-10, 10, 10)[:, np.newaxis]
y = np.sin((X[:, 0] - 2.5) ** 2)
obj_fun = lambda x: (x-0.5)*np.sin(x)
y = np.array([obj_fun(x) for x in X[:, 0]])
gp.fit(X, y)

# Plot posterior
plt.figure(figsize=(10,10))
ax = plt.subplot(1, 1, 1)
X_ = np.linspace(-10, 10, 500)

f =  np.array([obj_fun(x) for x in X_])

y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
plt.plot(X_, y_mean, 'k', lw=1, zorder=9)
plt.plot(X_, f, 'k--', lw=1, zorder=9)
plt.fill_between(X_, y_mean - 2*y_std, y_mean + 2*y_std,
                 alpha=0.1, color='k')

plt.scatter(X[:, 0], y, c='k', s=20, zorder=10)
plt.xlim(-10, 10)
plt.ylim(-10, 10)

plt.xticks([10], [r"$x$"], rotation=0, fontsize=30)
plt.yticks([], [], rotation=0, fontsize=30)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.tick_params(axis=u'both', which=u'both',length=0)

legend_elements = [Line2D([0], [0], linestyle = '--', color='k', lw=1, label='Objective'),
                   Line2D([0], [0], color='k', lw=1, label='Surrogate'),
                   Line2D([], [], marker='o', color='k', label='Observations', markerfacecolor='k', markersize=4)]
ax.legend(handles=legend_elements, loc='upper right', fancybox=True, framealpha=0.5, fontsize = 20)

plt.tight_layout()

#plt.show()
plt.savefig('surrogate.pdf', bbox_inches='tight')
