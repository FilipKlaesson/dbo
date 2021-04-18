import numpy as np
from matplotlib import cm
from scipy.stats import norm
from matplotlib import pyplot as plt

sigma = np.linspace(0,1,100)
delta = np.linspace(-1,1,100)

S, D = np.meshgrid(sigma, delta)

EI = D * norm.cdf(D/S) + S * norm.pdf(D/S)

plt.figure(figsize=(10,10))
ax = plt.subplot(1, 1, 1)
plt.contourf(S, D, EI, np.linspace(0,1.1,15), cmap = cm.coolwarm)

plt.xlim(0, 1)
plt.ylim(-1, 1)

plt.xticks([0, 0.25, 0.5, 0.75, 1], rotation=0, fontsize=15)
plt.yticks([-1, -0.5, 0, 0.5, 1], rotation=0, fontsize=15)

plt.xlabel(r"$\sigma_{n}(x)$", fontsize=30)
plt.ylabel(r"$\mu_{n}(x) - f^*_n$", rotation=90, fontsize=30)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#plt.show()
plt.savefig('ei.pdf', bbox_inches='tight')
