import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm

files = sys.argv[1::2]
ids = sys.argv[2::2]

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
plt.yscale('log')

for file, id in zip(files, ids):

    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        regret = []
        regret_err = []
        dist = []
        for row in reader:
            regret.append(max(0, float(row[1])))
            regret_err.append(float(row[2]))
            dist.append(10**(-2)*float(row[3]))

        # relative regret_error for log scale
        lower = [10**(np.log10(m) + (0.434*e/m)) for m, e in zip(regret, regret_err)]
        upper = [10**(np.log10(m) - (0.434*e/m)) for m, e in zip(regret, regret_err)]

        id = '$' + id + '$'

        plt.plot(dist, regret, '-', linewidth=1, label = id)
        plt.fill_between(dist, upper, lower, alpha=0.3)

trans = ax.get_xaxis_transform()
ann = ax.annotate('$\\times 10^{2}$', xy=(20.6, -0.01 ), xycoords=trans, fontsize=10)

plt.legend(prop={'size': 12}, loc = 'upper right')
plt.xlabel('distance', fontsize=12, rotation=0)
plt.ylabel('regret R', fontsize=12, rotation=90)
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.6)
plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.3)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('distance-regret.pdf', bbox_inches='tight')
plt.savefig('distance-regret.png', bbox_inches='tight')
