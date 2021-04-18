import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.1)
gold = 0.1*(x-1.5)*np.sin(x)-2
land = 0*x

plt.fill_between(x, -3, gold, color = 'gold')
plt.fill_between(x, gold, land, color = 'sandybrown')
plt.plot(x, land, c = 'k')

plt.savefig('gold.png', bbox_inches='tight')
