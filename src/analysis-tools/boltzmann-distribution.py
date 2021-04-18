import numpy as np
from matplotlib import cm
from scipy.stats import norm
from matplotlib import pyplot as plt
import scipy.stats as stats
import math
from matplotlib.lines import Line2D

N = 100

def softmax(inputs):
    return np.exp(inputs) / (np.exp(1)-1)

x = np.linspace(0, 1, N)
b = softmax(x)

plt.figure(figsize=(10,10))
ax = plt.subplot(1, 1, 1)

plt.plot(x, b, 'k-')

plt.xlim(0, 1)

plt.xlabel(r"$Q$", fontsize=30)
plt.ylabel(r"$p_b(Q)$", rotation=0, fontsize=30, labelpad=30)
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=0)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.grid(True)
#plt.show()
plt.savefig('boltzmann.pdf', bbox_inches='tight')
