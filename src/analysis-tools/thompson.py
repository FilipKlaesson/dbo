import numpy as np
from matplotlib import cm
from scipy.stats import norm
from matplotlib import pyplot as plt
import scipy.stats as stats
import math
from matplotlib.lines import Line2D

plt.figure(figsize=(10,10))
ax = plt.subplot(1, 1, 1)

mu = 0.6
variance = 0.001
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'b-')

mu = 0.4
variance = 0.002
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-')

mu = 0.5
variance = 0.02
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'g-')


plt.xlim(0, 1)
plt.ylim(0, 20)

plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1], rotation=0, fontsize=15)
plt.yticks([0, 5, 10, 15, 20], rotation=0, fontsize=15)

plt.xlabel(r"$r$", fontsize=30)
plt.ylabel(r"$\phi(r)$", rotation=0, fontsize=30, labelpad=30)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

legend_elements = [Line2D([0], [0], color='b', lw=1, label='Action 1'),
                   Line2D([0], [0], color='r', lw=1, label='Action 2'),
                   Line2D([0], [0], color='g', lw=1, label='Action 3')]
ax.legend(handles=legend_elements, loc='upper right', fancybox=True, framealpha=0.5, fontsize = 20)

#plt.show()
plt.savefig('thompson.pdf', bbox_inches='tight')
