import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
from bayesian_optimization import bayesian_optimization

# Domain
domain = np.array([[-10, 10]])

# Objective function
obj_fun = lambda x: (x[0]-0.5)*np.sin(x[0])

# Communication network
num_agents = 1
N = np.eye(3)
N[0,1] = N[1,0] = N[1,2] = N[2,1] = 1

#Inital points
init_data = [[1], [2], [3]]

iter = 50
iterations = 10
SR_list = [[] for i in range(iterations)]
for i in range(iterations):

    # Bayesian optimimzation object
    BO = bayesian_optimization( obj = obj_fun, domain = domain,
                                n_workers = num_agents, network = None,
                                kernel = gp.kernels.RBF(length_scale_bounds=(1, 1000)),
                                acquisition_function = 'ei',
                                stochastic_policy = False,
                                regularization = None, l = 0.01,
                                grid_density = 10000)

    # Optimize
    BO.optimize(n_iters = iter, x0 = init_data, n_pre_samples = 3, random_search = 100, plot = False)

    SR_list[i].append(BO.simple_regret)

SR = []
SR_std = []
for t in range(iter):
        SR.append(np.mean([sr[t] for i in range(iterations) for sr in SR_list[i]]))
        SR_std.append(np.std([sr[t] for i in range(iterations) for sr in SR_list[i]]))

for i in range(len(SR_std)):
    SR_std[i] = 1.96*SR_std[i]/iterations

SR1 = [10**(np.log10(sr_i) + (0.434*sr_std_i/sr_i)) for sr_i, sr_std_i in zip(SR, SR_std)]
SR2 = [10**(np.log10(sr_i) - (0.434*sr_std_i/sr_i)) for sr_i, sr_std_i in zip(SR, SR_std)]

SR = [sr_i for sr_i in SR]

fig = plt.figure()
plt.yscale('log')
plt.plot(range(iter), SR, '-')
plt.fill_between(range(iter), SR1, SR2, alpha=0.3)
plt.grid(True)
plt.show()
