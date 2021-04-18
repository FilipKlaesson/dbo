import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ExpSineSquared)


kernel = RBF(length_scale=0.5, length_scale_bounds='fixed') #ExpSineSquared(length_scale=1.0, periodicity=1.0, length_scale_bounds='fixed', periodicity_bounds='fixed')
gp = GaussianProcessRegressor(kernel=kernel)
samples = 5

# Plot prior
plt.figure(figsize=(10,10))
ax = plt.subplot(1, 1, 1)
X_ = np.linspace(-5, 5, 500)
y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
plt.plot(X_, y_mean, 'k--', lw=1, zorder=9)
plt.fill_between(X_, y_mean - 2*y_std, y_mean + 2*y_std,
                 alpha=0.1, color='k')
y_samples = gp.sample_y(X_[:, np.newaxis], samples)
plt.plot(X_, y_samples, lw=1)
plt.xlim(-5, 5)
plt.ylim(-3, 3)
#plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

plt.xticks([5], [r"$x$"], rotation=0, fontsize=30)
plt.yticks([3], [r"$f(x)$"], rotation=0, fontsize=30)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.tick_params(axis=u'both', which=u'both',length=0)

# Generate data and fit GP
rng = np.random.RandomState(11)
X = rng.uniform(-4, 4, 4)[:, np.newaxis]
y = np.sin((X[:, 0] - 2.5) ** 2)
gp.fit(X, y)

#plt.show()
plt.savefig('prior-rbf-0.5.pdf', bbox_inches='tight')

# Plot posterior
plt.figure(figsize=(10,10))
ax = plt.subplot(1, 1, 1)
X_ = np.linspace(-5, 5, 500)
y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
plt.plot(X_, y_mean, 'k', lw=1, zorder=9)
plt.fill_between(X_, y_mean - 2*y_std, y_mean + 2*y_std,
                 alpha=0.1, color='k')

y_samples = gp.sample_y(X_[:, np.newaxis], samples)
plt.plot(X_, y_samples, lw=1)
plt.scatter(X[:, 0], y, c='k', s=20, zorder=10)
plt.xlim(-5, 5)
plt.ylim(-3, 3)
#plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
#          % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)),
#          fontsize=12)


plt.xticks([5], [r"$x$"], rotation=0, fontsize=30)
plt.yticks([3], [r"$f(x)$"], rotation=0, fontsize=30)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.tick_params(axis=u'both', which=u'both',length=0)

plt.tight_layout()

#plt.show()
plt.savefig('posterior-rbf-0.5.pdf', bbox_inches='tight')
