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
plt.yscale('log')

for file, id in zip(files, ids):

    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        iter = []
        mean = []
        err = []
        for row in reader:
            iter.append(int(row[0]))
            mean.append(max(0, float(row[1])))
            err.append(float(row[2]))

        # relative error for log scale
        lower = [10**(np.log10(m) + (0.434*e/m)) for m, e in zip(mean, err)]
        upper = [10**(np.log10(m) - (0.434*e/m)) for m, e in zip(mean, err)]

        id = '$' + id + '$'

        plt.plot(iter, mean, '-', linewidth=1, label = id)
        plt.fill_between(iter, upper, lower, alpha=0.3)

plt.legend(prop={'size': 12}, loc = 'upper right')
plt.xlabel('iteration', fontsize=12, rotation=0)
plt.ylabel('regret R', fontsize=12, rotation=90)
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.6)
plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.3)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('log-regret.pdf', bbox_inches='tight')
plt.savefig('log-regret.png', bbox_inches='tight')
